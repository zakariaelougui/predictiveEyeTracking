import json
import base64
import requests
import gzip
from myscript import main
from predictiveEyeTracking import generate_base64_heatmap

def lambda_handler(event, context):
    body = json.loads(event['body'])

    if 'image' not in body:
        return {
            'statusCode': 400,
            'body': json.dumps({'message': 'image field is required'})
        }
    # Step 1: Decode the Base64 string to get the compressed binary data
    # compressed_data = base64.b64decode(body.get('image'))
    imageReceived = download_image_as_base64(body['image'])

    image = generate_base64_heatmap(imageReceived, body.get('color'), body.get('min_points'), body.get('max_points'), body.get('show_fixation_points'))

    return {
        'statusCode': 200,
        'body': json.dumps({'image': image})
    }

def download_image_as_base64(image_url):
    try:
        # Send a GET request to the image URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Convert image content to Base64
        image_data = response.content
        base64_encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        return base64_encoded_image

    except requests.RequestException as e:
        print(f'Error downloading image: {e}')
        return None