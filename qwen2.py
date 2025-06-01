import json
import os

import ollama
import pytesseract
from PIL import Image

# Path to the folder containing images
image_folder = './image_output2'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Function to extract text from images using Tesseract OCR
def extract_text_from_image(image_path):
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang='tam')  # Tamil language support
        return text.strip()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Extract text from all images
extracted_data = []
for img_file in image_files:
    image_path = os.path.join(image_folder, img_file)
    text = extract_text_from_image(image_path)
    if text:
        extracted_data.append({"image": img_file, "text": text})


# List to store all responses
all_data_list = []

# Format the data for the DeepSeek model
if extracted_data:
    formatted_text = "\n\n".join(
        [
            f"Image: {data['image']}\nExtracted Text:\n{data['text']}"
            for data in extracted_data
        ]
    )
    
    # Formulate the prompt for DeepSeek
    prompt = f"""
    The following are extracted texts from a voters list in Tamil:
    {formatted_text}

    Note: Don't ever translate tamil to english, keep tamil words as it is

    Analyse this image and give a array of objects response, here there are details about each person in a voters list, each box denotes a person details like name, age, sex, door no, father's name also sometimes husband name, voterID at top right of each box with 10 digit character, all other these details will be in tamil and id at top left  it will have only numbers in it with 3 or 4 characters. IF data isn't available for specific key use null as value, if data is na then use null as value, extract all the persons data in a array of objects response of all, if text is similar to கணவர் பெயர் means husband's name, if text is similar to தந்தை பெயர் it means father's name, so next time when you generate keep this in mind, if the image doesn't have the relevant information skip that image, don't generate any response, also don't even response with any message just return empty string

    """

    # Call DeepSeek via Ollama
    response = ollama.chat(model='mistral:latest', messages=[
        {'role': 'system', 'content': 'You are a helpful assistant. You are going to help me to find few persons voter details for attached image, without translating just extract the text accurately'},
        {'role': 'user', 'content': prompt}
    ])

    # Output the response
    print(response)

    # Clean the response text
    cleaned_response_text = response['message']['content'].replace("```json", "").replace("```", "").strip()

    if cleaned_response_text:
        try:
            response_data = json.loads(cleaned_response_text)
            if isinstance(response_data, list):
                all_data_list.extend(response_data)

            # Create a dictionary with the key 'data' and the accumulated list
            all_data = {"data": all_data_list}

            # Write all responses to a single JSON file
            with open('mistral.json', 'w') as json_file:
                json.dump(all_data, json_file, indent=4, ensure_ascii=False)

            print('All responses written to all_responses.json')
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
else:
    print("No valid text extracted from images.")
