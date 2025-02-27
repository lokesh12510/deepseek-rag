import json
import os

import PIL.Image
from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel

load_dotenv()

class VoterModel(BaseModel):
    id: str
    name: str
    fathers_name: str
    husbands_name: str
    gender: str
    age: str
    voter_id: str
    door_no: str

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

# Directory containing the images
image_dir = './image_output'

# Initialize the Vision API client
client = genai.Client(api_key=GOOGLE_API_KEY)

# List to store all responses
all_data_list = []

# Loop over all images in the directory
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        image = PIL.Image.open(image_path)
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["Analyse this image and give a array of objects response, here there are details about each person in a voters list, each box denotes a person details like name, age, sex, door no, father's name also sometimes husband name, voterID at top right of each box with 10 digit character, all other these details will be in tamil and id at top left  it will have only numbers in it with 3 or 4 characters. IF data isn't available for specific key use null as value, if data is na then use null as value, extract all the persons data in a array of objects response of all, if text is similar to கணவர் பெயர் means husband's name, if text is similar to தந்தை பெயர் it means father's name, so next time when you generate keep this in mind, if the image doesn't have the relevant information skip that image, don't generate any response, also don't even response with any message just return empty string", image],
        )

        # Clean the response text
        cleaned_response_text = response.text.replace("```json", "").replace("```", "").strip()

        # Parse the cleaned response text as JSON and add to all_data_list
        if cleaned_response_text:
            try:
                response_data = json.loads(cleaned_response_text)
                if isinstance(response_data, list):
                    all_data_list.extend(response_data)
                else:
                    print(f"Unexpected response format for image {image_file}")
            except json.JSONDecodeError:
                print(f"Failed to decode JSON for image {image_file}")

# Create a dictionary with the key 'data' and the accumulated list
all_data = {"data": all_data_list}

# Write all responses to a single JSON file
with open('all_responses_3.json', 'w', encoding='utf-8') as json_file:
    json.dump(all_data, json_file, indent=4, ensure_ascii=False)

print('All responses written to all_responses.json')