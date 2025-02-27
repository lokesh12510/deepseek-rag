import os

import PIL.Image
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
from pydantic import BaseModel


class VoterModel(BaseModel):
     id:str
     name:str
     fathers_name:str
     husbands_name:str
     gender:str
     age:str
     voter_id:str
     door_no:str

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")

image = PIL.Image.open('./image_output/output_page_1.png')

client = genai.Client(api_key=GOOGLE_API_KEY)
response  = client.models.generate_content(
     model="gemini-2.0-flash",
     contents=["Analyse this image and give a array of objects response, here there are details about each person in a voters list, each box denotes a person details like name, age, sex, door no, father's name also sometimes husband name, id at top left and voter ID  at top right, all these details will be in tamil. IF data isn't available for specific key use null, if data is na also use null, extract all the persons data in a array of objects response of all, if text is similar to கணவர் பெயர் means husband's name, if text is similar to தந்தை பெயர் it means father's name, so next time when you generate keep this in mind, if the image doesn't have the relevant information skip that image,don't generate any response, also don't even response with any message just return empty string", image],
)


# Write the response text to a file
with open('response_1.txt', 'w', encoding='utf-8') as file:
    file.write(response.text)

print('Response written to response.txt')