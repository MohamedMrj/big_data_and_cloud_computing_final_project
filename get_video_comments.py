# Import the necessary modules
import requests
from creds import API_KEY, VIDEO_ID  # Import API_KEY and VIDEO_ID from creds module
from api import get_response_data

def video_comments():
    comments = []
    next_page_token = None
    
    while True:
        try:
            data = get_response_data(VIDEO_ID, next_page_token)
            if 'items' in data:
                for item in data['items']:
                    comments.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
                
                next_page_token = data.get("nextPageToken")
                if not next_page_token:
                    break  # No more pages to fetch
            else:
                print("No items in the response")
                break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    return comments