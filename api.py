from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from Skinlytics.backend.app import create_app

app = create_app()
import json

def handler(event, context):
    # This function will be called by Vercel's serverless function
    return {
        'statusCode': 200,
        'headers': {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        },
        'body': json.dumps({'message': 'API is working!'})
    }
