import boto3
import os
import pandas as pd
import awsgi
from flask import (
    Flask,
    render_template, request
)

from jinja2 import Environment, FileSystemLoader

# Load Jinja2 template environment
env = Environment(loader=FileSystemLoader('.'))
app = Flask(__name__)
s3 = boto3.client('s3')
bucket_name = os.environ['bucket_name']
file_key = os.environ['file_key']

def lambda_handler(event, context):
    # Get the object from the S3 bucket
    response = s3.get_object(Bucket=bucket_name, Key=file_key)

    # Read the CSV file directly into a DataFrame
    df = pd.read_csv(response['Body'])
    hospital_names = df['Org Name'].unique()

    # Get the hospital filter parameter from the event data
    filter_hospital = event.get('hospital')

    # Filter the data based on the selected hospital
    if filter_hospital:
        df = df[df['Org Name'] == filter_hospital]


    # Convert DataFrame to list of dictionaries
    comments = df.to_dict(orient='records')

    # Render HTML response using Jinja2 template
    template = env.get_template('index.html')
    html = template.render(comments=comments, hospital_names=hospital_names, selected_hospital=filter_hospital)

    return html

# def lambda_handler(event, context):
#     return awsgi.response(app, event, context)
