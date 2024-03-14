# Import the necessary modules
import requests
from creds import API_KEY  # Import API_KEY from creds module

# Define a function to get video comments
# Define a function to get video comments
def get_response_data(video_id, page_token=None):
    # Construct the initial URL
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&part=snippet&videoId={video_id}&maxResults=100"
    
    if page_token:
        # If a page token is provided, include it in the request URL
        url += f"&pageToken={page_token}"

    response = requests.get(url)
    data = response.json()
    return data