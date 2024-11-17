import os
import requests
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_oauth_token():
    """Get OAuth token for authentication"""
    client_id = os.getenv('SENTINEL_CLIENT_ID')
    client_secret = os.getenv('SENTINEL_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        raise ValueError("Missing SENTINEL_CLIENT_ID or SENTINEL_CLIENT_SECRET in .env file")
    
    token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    
    response = requests.post(token_url,
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    )
    
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception(f"Failed to get token: {response.text}")

def download_sentinel_images(start_date="2023-09-01", end_date="2023-11-30"):
    """Download Sentinel-1 GRD images for specified period"""
    print(f"Downloading Sentinel-1 images from {start_date} to {end_date}")
    
    token = get_oauth_token()
    
    # Sonderborg bounding box coordinates (from old utils)
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [9.577693, 54.829313],
            [10.013713, 54.829313],
            [10.013713, 54.967015],
            [9.577693, 54.967015],
            [9.577693, 54.829313]
        ]]
    }
    
    process_url = "https://sh.dataspace.copernicus.eu/api/v1/process"
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    # Using the SAR evalscript from old utils with modifications
    evalscript = """
    //VERSION=3
    function setup() {
        return {
            input: [{
                bands: ["VV", "VH"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.VV, sample.VH, (sample.VV + sample.VH) / 2];
    }
    """

    process_body = {
        "input": {
            "bounds": {
                "geometry": geometry,
                "properties": {"crs": "http://www.opengis.net/def/crs/EPSG/0/4326"}
            },
            "data": [{
                "type": "sentinel-1-grd",
                "dataFilter": {
                    "timeRange": {
                        "from": f"{start_date}T00:00:00Z",
                        "to": f"{end_date}T23:59:59Z"
                    },
                    "orbitDirection": "ASCENDING"
                }
            }]
        },
        "output": {
            "width": 1024,
            "height": 1024,
            "responses": [{
                "identifier": "default",
                "format": {"type": "image/tiff"}
            }]
        },
        "evalscript": evalscript
    }

    # Create output directory
    output_dir = "data/sonderborg/sentinel-1/raw"
    os.makedirs(output_dir, exist_ok=True)

    response = requests.post(process_url, headers=headers, json=process_body)
    
    if response.status_code == 200:
        # Save the image with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"sentinel1_grd_{timestamp}.tiff")
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print(f"Image saved to: {output_path}")
    else:
        print(f"Failed to process request: {response.status_code} - {response.text}")

if __name__ == "__main__":
    download_sentinel_images()