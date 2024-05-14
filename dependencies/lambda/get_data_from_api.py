import json
import os
import boto3
import urllib.request
from datetime import datetime

def lambda_handler(event, context):
    # Fetch URL, params, and headers from environment variables
    url = os.environ.get("url")
    params = json.loads(os.environ.get("params", "{}"))
    headers = json.loads(os.environ.get("headers", "{}"))

    # Retrieve data from URL
    data = retrieve_data_from_url(url, params, headers)
    print(data)

    if data:
        # Save data to S3
        bucket_name = "dyploma-yuliia"  # Update with your S3 bucket name
        save_data_to_s3(data, bucket_name)

def retrieve_data_from_url(url, params, headers):
    # Construct the URL with query parameters
    url_with_params = f"{url}?{'&'.join([f'{key}={value}' for key, value in params.items()])}"

    try:
        # Construct a request object with headers
        request = urllib.request.Request(url_with_params, headers=headers)
        # Retrieve data from the URL
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode('utf-8'))
        return data
    except Exception as e:
        print('Error:', e)
        return None

def save_data_to_s3(data, bucket_name):
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{current_datetime}.json"
    s3 = boto3.client('s3')
    s3_key = f"Datalake/raw-data/{filename}"

    s3.put_object(Bucket=bucket_name, Key=s3_key, Body=json.dumps(data))

    print(f"Data saved to s3://{bucket_name}/{s3_key}")
