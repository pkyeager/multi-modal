# This script is currently not working as expected.



import os 
from datetime import datetime
from dotenv import load_dotenv
from sentinelhub import (
    SHConfig,
    CRS,
    BBox,
    DataCollection,
    MimeType,
    SentinelHubRequest,
    SentinelHubCatalog,
    SentinelHubDownloadClient
)
from shapely.geometry import mapping

# Load environment variables
load_dotenv('.env.local')

def setup_config():
    # Get environment variables
    client_id = os.getenv('SH_CLIENT_ID')
    client_secret = os.getenv('SH_CLIENT_SECRET')
    
    # Validate credentials
    if not client_id or not client_secret:
        raise ValueError(
            "Missing credentials. Please ensure SH_CLIENT_ID and SH_CLIENT_SECRET "
            "are set in your .env.local file"
        )
    
    # Configure Sentinel Hub
    config = SHConfig()
    config.sh_client_id = client_id
    config.sh_client_secret = client_secret
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    
    if not config.sh_client_id or not config.sh_client_secret:
        raise ValueError(
            "Configuration error. Please check your credentials in .env.local file"
        )
    
    return config

def get_sentinel_images(location, start_date, end_date, output_folder):
    # Setup and validate configuration
    try:
        config = setup_config()
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        return
    
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    sonderborg_bbox = BBox((9.577693, 54.829313, 10.013713, 54.967015), crs=CRS.WGS84)
    
    try:
        catalog = SentinelHubCatalog(config=config)
        
        # Define the search parameters using CQL2
        cql2_filter = {
            "op": "and",
            "args": [
                {
                    "op": "s_intersects",
                    "args": [
                        {
                            "property": "geometry"
                        },
                        {
                            "type": "Polygon",
                            "coordinates": [[
                                [9.577693, 54.829313],
                                [9.577693, 54.967015],
                                [10.013713, 54.967015],
                                [10.013713, 54.829313],
                                [9.577693, 54.829313]
                            ]]
                        }
                    ]
                },
                {
                    "op": "t_intersects",
                    "args": [
                        {
                            "property": "datetime"
                        },
                        {
                            "interval": [
                                start_date,
                                end_date
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Search for Sentinel-1 images
        search_iterator = catalog.search(
            collection=DataCollection.SENTINEL1,
            filter=cql2_filter,
            fields={"include": ["id", "properties.datetime"], "exclude": []},
            filter_lang="cql2-json"
        )
        
        results = list(search_iterator)
        print(f"Total number of Sentinel-1 results:", len(results))
        
        if len(results) == 0:
            print("No images found. Please check your search parameters and date range.")
            return
        
        # Define the evalscript for Sentinel-1
        evalscript_sar = """
            //VERSION=3
            function setup() {
                return {
                    input: ["VH", "dataMask"],
                    output: [
                        { id: "default", bands: 4 },
                        { id: "eobrowserStats", bands: 1 },
                        { id: "dataMask", bands: 1 },
                    ],
                };
            }

            function evaluatePixel(samples) {
                const value = Math.max(0, Math.log(samples.VH) * 0.21714724095 + 1);
                return {
                    default: [value, value, value, samples.dataMask],
                    eobrowserStats: [Math.max(-30, (10 * Math.log10(samples.VH)))],
                    dataMask: [samples.dataMask],
                };
            }
        """
        
        # Download all images
        for result in results:
            try:
                # Set up the request for each image
                request = SentinelHubRequest(
                    evalscript=evalscript_sar,
                    input_data=[
                        SentinelHubRequest.input_data(
                            data_collection=DataCollection.SENTINEL1,
                            time_interval=(result.properties['datetime'], result.properties['datetime']),
                        )
                    ],
                    responses=[
                        SentinelHubRequest.output_response("default", MimeType.TIFF),
                        SentinelHubRequest.output_response("eobrowserStats", MimeType.TIFF),
                        SentinelHubRequest.output_response("dataMask", MimeType.TIFF)
                    ],
                    bbox=sonderborg_bbox,
                    config=config,
                )
                
                # Get the data
                data = request.get_data()
                
                # Process timestamp
                timestamp = result.properties['datetime'].replace(':', '-')
                
                # Save all three image types
                for idx, (name, band_data) in enumerate(zip(['default', 'stats', 'mask'], data)):
                    filename = f"S1_{name}_{timestamp}.tiff"
                    filepath = os.path.join(output_folder, filename)
                    with open(filepath, 'wb') as f:
                        f.write(band_data)
                    print(f"Saved {name} image to {filepath}")
                
            except Exception as e:
                print(f"Error processing image {result.properties['datetime']}: {str(e)}")
                continue
            
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return

if __name__ == "__main__":
    # Ensure the .env.local file exists
    if not os.path.exists('.env.local'):
        print("Error: .env.local file not found. Please create it with your credentials.")
        exit(1)
        
    location = "Sonderborg, Denmark"
    start_date = "2023-10-01"
    end_date = "2023-10-31"
    
    # Fetch Sentinel-1 images
    get_sentinel_images(location, start_date, end_date, "fetched_data/sentinel1")
