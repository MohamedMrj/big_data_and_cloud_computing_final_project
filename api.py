# Import the necessary modules
import requests
from creds import API_KEY  # Import API_KEY from creds module

# Define a function to get video comments
def get_response_data(video_id):
    # Construct the URL for the API request
    url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&part=snippet&videoId={video_id}&maxResults=100'
    
    # Start a loop to fetch comments
    while True:
        # Send a GET request to the YouTube API
        response = requests.get(url)
        
        # Convert the response to JSON
        data = response.json()
        
        # If 'nextPageToken' is in the response data, update the url with the new page token
        if "nextPageToken" in data:
            url = f'https://www.googleapis.com/youtube/v3/commentThreads?key={API_KEY}&part=snippet&videoId={video_id}&pageToken={data["nextPageToken"]}&maxResults=100'
        else:
            # If not, return the data as there are no more comments to fetch
            return data