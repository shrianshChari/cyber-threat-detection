import gc
import os
import time
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator
import clickhouse_connect
from pyarrow.fs import S3FileSystem, FileType
import polars as pl
from dotenv import load_dotenv
import boto3

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
OBJECT_KEY = os.getenv("OBJECT_KEY")
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "us-west-1")

def get_aws_credentials():
    session = boto3.Session(region_name=AWS_REGION)
    credentials = session.get_credentials().get_frozen_credentials()
    return credentials

def get_s3_storage_options():
    credentials = get_aws_credentials()
    storage_options = {
        "aws_access_key_id": credentials.access_key,
        "aws_secret_access_key": credentials.secret_key,
        "aws_session_token": credentials.token,
        "aws_region": AWS_REGION,
    }

    endpoint_url = S3_ENDPOINT = os.getenv("S3_ENDPOINT_URL", "").strip()
    if endpoint_url.startswith(('http://', 'https://')):
        storage_options["aws_endpoint_url"] = endpoint_url

    return storage_options

CH_HOST = os.getenv("CLICKHOUSE_HOST", "localhost")
CH_PORT = int(os.getenv("CLICKHOUSE_PORT", 8123))
CH_USER = os.getenv("CLICKHOUSE_USER", "admin")
CH_PASS = os.getenv("CLICKHOUSE_PASSWORD", "password")

def get_clickhouse_client():
    client = clickhouse_connect.get_client(
        host=CH_HOST,
        port=CH_PORT,
        username=CH_USER,
        password=CH_PASS,
        send_receive_timeout=3600
    )

    client.set_client_setting("max_memory_usage", 0)
    client.set_client_setting("async_insert", 0)
    client.set_client_setting('max_insert_block_size', 50000)

    return client

def task_check_s3_file():
    """Verify source Parquet file exists in S3/MinIO."""
    storage_options = get_s3_storage_options()

    s3_url = f's3://{BUCKET_NAME}/{OBJECT_KEY}'

    print(f"Scanning S3 file at {s3_url} via Polars...")
    lf = pl.scan_parquet(s3_url, storage_options=storage_options)

    total_rows = lf.select(pl.len()).collect().item()
    print(
        f"Successfully connected to S3! File contains {total_rows:,} records."
    )

def task_init_schema():
    """Auto-create ClickHouse target table schema if it does not exist."""
    client = get_clickhouse_client()

    # Generate S3 function URL for schema inference
    creds = get_aws_credentials()
    s3_url = f"https://s3.{AWS_REGION}.amazonaws.com/{BUCKET_NAME}/{OBJECT_KEY}"
    s3_function = f"s3('{s3_url}', '{creds.access_key}', '{creds.secret_key}', '{creds.token}', 'Parquet')"

    sql = f"""
    CREATE OR REPLACE TABLE default.logs_data
    ENGINE = MergeTree()
    ORDER BY tuple()
    EMPTY AS 
    SELECT * FROM {s3_function};
    """
    client.command(sql)
    print("Table 'default.logs_data' schema initialized.")

def task_stream_ingest():
    """Stream Parquet batches from S3 into ClickHouse to maintain low RAM usage."""
    client = get_clickhouse_client()

    storage_options = get_s3_storage_options()

    s3_url = f"s3://{BUCKET_NAME}/{OBJECT_KEY}"

    lf = pl.scan_parquet(s3_url, storage_options=storage_options)

    total_rows = lf.select(pl.len()).collect().item()
    print(f"Dataset contains {total_rows:,} total records.")

    BATCH_SIZE = 200_000
    rows_inserted = 0
    start_time = time.time()

    for offset in range(0, total_rows, BATCH_SIZE):
        # Fetch slice lazily; Polars decodes only the row groups needed for this offset
        df_chunk = lf.slice(offset, BATCH_SIZE).collect()
        
        # Convert chunk to PyArrow table for zero-copy streaming into ClickHouse
        arrow_table = df_chunk.to_arrow()
        client.insert_arrow("default.logs_data", arrow_table)
        
        rows_inserted += len(df_chunk)
        print(f"Ingested {rows_inserted:,} / {total_rows:,} rows ({(rows_inserted/total_rows)*100:.1f}%)")

        # Clean up memory buffers immediately
        del df_chunk, arrow_table
        gc.collect()

    print(f"Polars streaming ingestion finished in {time.time() - start_time:.2f} seconds.")

def task_validate():
    """Validate total row count in ClickHouse."""
    client = get_clickhouse_client()
    count = client.command("SELECT count() FROM default.logs_data")
    print(f"Ingestion check successful. Total records in default.logs_data: {count:,}")

# DAG Definition
with DAG(
    dag_id="manual_s3_to_clickhouse_ingestion",
    start_date=datetime(2026, 10, 3),
    schedule=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    check_s3 = PythonOperator(
        task_id="check_s3_file",
        python_callable=task_check_s3_file,
    )

    init_schema = PythonOperator(
        task_id="initialize_clickhouse_schema",
        python_callable=task_init_schema,
    )

    stream_ingest = PythonOperator(
        task_id="stream_parquet_to_clickhouse",
        python_callable=task_stream_ingest,
    )

    validate = PythonOperator(
        task_id="validate_ingestion",
        python_callable=task_validate,
    )

    check_s3 >> init_schema >> stream_ingest >> validate
