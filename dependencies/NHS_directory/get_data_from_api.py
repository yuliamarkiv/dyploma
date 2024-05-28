import requests
from datetime import datetime
import json

# Set up authentication if required
headers = {'subscription-key': ''}

# Construct the URL for the desired endpoint with parameters
url = 'https://api.nhs.uk/comments/Comments'
params = {'orgType': 'HOS', 'limit': 100}

# Send the GET request to the API
response = requests.get(url, params=params, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the response data
    data = response.json()
    # Generate filename with current datetime
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data_{current_datetime}.json"

    # Save data to file
    with open(filename, 'w') as file:
        file.write(json.dumps(data))

    print(f"Data saved to {filename}")
else:
    # Handle errors
    print('Error:', response.status_code)
