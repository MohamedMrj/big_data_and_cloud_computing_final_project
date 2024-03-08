# Import the necessary modules
import requests
from creds import API_KEY, VIDEO_ID  # Import API_KEY and VIDEO_ID from creds module
from api import get_response_data

# Define a function to get video comments
def video_comments():
    try:
        data = get_response_data(VIDEO_ID)
        comments = []
        # Check if 'items' is in the response data
        if 'items' in data:
            # Loop through each item in 'items'
            for item in data['items']:
                # Append the comment text to the comments list
                comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                
            # Check if 'nextPageToken' is in the response data
            if "nextPageToken" in data:
                # If so, set next_page_token to the value of 'nextPageToken'
                next_page_token = data["nextPageToken"]
            else:
                # If not, return the comments as there are no more comments to fetch
                return comments
        else:
            # If 'items' is not in the response data, print a message and return an empty list
            print("No items in the response")
            return comments
    except Exception as e:
        # If an exception occurs, print the error message and return an empty list
        print(f"An error occurred: {e}")
        return []