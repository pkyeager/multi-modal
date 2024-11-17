import os
from datetime import datetime, timedelta
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

# Load environment variables
load_dotenv('.env.local')

def setup_config():
    config = SHConfig()
    config.sh_client_id = os.getenv('SH_CLIENT_ID')
    config.sh_client_secret = os.getenv('SH_CLIENT_SECRET')
    config.sh_token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    config.sh_base_url = "https://sh.dataspace.copernicus.eu"
    return config

def get_sentinel_images(location, start_date, end_date, product_type, output_folder):
    config = setup_config()
    
    # Define the bounding box for Sonderborg
    sonderborg_bbox = BBox((9.577693, 54.829313, 10.013713, 54.967015), crs=CRS.WGS84)
    
    # Set up the catalog
    catalog = SentinelHubCatalog(config=config)
    
    # Define the search parameters using CQL2
    cql2_filter = {
        "op": "and",
        "args": [
            {"op": "s_intersects", "args": [{"property": "geometry"}, sonderborg_bbox.geometry]},
            {"op": ">=", "args": [{"property": "datetime"}, start_date]},
            {"op": "<=", "args": [{"property": "datetime"}, end_date]}
        ]
    }
    
    # Define the search parameters
    search_iterator = catalog.search(
        collection=DataCollection[product_type],
        filter=cql2_filter,
        fields={"include": ["id", "properties.datetime", "properties.eo:cloud_cover"], "exclude": []}
    )
    
    results = list(search_iterator)
    print(f"Total number of {product_type} results:", len(results))
    
    if len(results) == 0:
        print("No images found. Please check your search parameters and date range.")
        return
    
    # Define the evalscript for true color images (for Sentinel-2)
    evalscript_true_color = """
        //VERSION=3
        function setup() {
            return {
                input: [{
                    bands: ["B02", "B03", "B04"]
                }],
                output: {
                    bands: 3
                }
            };
        }
        function evaluatePixel(sample) {
            return [sample.B04, sample.B03, sample.B02];
        }
    """
    
    # Define the evalscript for Sentinel-1 GRD
    evalscript_sar = """
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
    
    # Download all images
    for result in results:
        # Set up the request for each image
        request = SentinelHubRequest(
            evalscript=evalscript_true_color if product_type == "SENTINEL2_L2A" else evalscript_sar,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection[product_type],
                    time_interval=(result.properties['datetime'], result.properties['datetime']),
                )
            ],
            responses=[SentinelHubRequest.output_response("default", MimeType.TIFF)],
            bbox=sonderborg_bbox,
            config=config,
        )
        
        # Get the data
        data = request.get_data()
        
        # Save the image
        os.makedirs(output_folder, exist_ok=True)
        image_filename = f"{product_type}_{result.properties['datetime']}.tiff"
        image_path = os.path.join(output_folder, image_filename)
        with open(image_path, 'wb') as f:
            f.write(data[0])
        
        print(f"Downloaded image saved to {image_path}")

if __name__ == "__main__":
    location = "Sonderborg, Denmark"
    start_date = "2023-10-01"
    end_date = "2023-10-31" 

    # Fetch Sentinel-2 L2A images
    get_sentinel_images(location, start_date, end_date, "SENTINEL2_L2A", "fetched_data/sentinel2")

    # Fetch Sentinel-1 GRD images
    get_sentinel_images(location, start_date, end_date, "SENTINEL1_GRD", "fetched_data/sentinel1")
