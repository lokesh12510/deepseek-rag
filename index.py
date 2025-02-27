import io
import json
import os

from google.cloud import vision
from google.cloud.vision_v1 import types
from pdf2image import convert_from_path

# Path to the PDF file
pdf_path = './ac118001.pdf'

# Convert PDF to images with high quality
images = convert_from_path(pdf_path, dpi=300)

# Initialize the Vision API client
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'path/to/your/service-account-file.json'
client = vision.ImageAnnotatorClient()

# Function to perform OCR on an image and return the text
def extract_text_from_image(image):
    content = io.BytesIO()
    image.save(content, format='PNG')
    content = content.getvalue()
    image = types.Image(content=content)
    response = client.text_detection(image=image)
    texts = response.text_annotations
    return texts[0].description if texts else ''

# Hypothetical function to add additional context using an AI model
def add_context_to_text(text):
    # This is a placeholder for the actual implementation of the AI model
    # For example, you could use a pre-trained language model to add context
    return f"Contextualized: {text}"

# Save each page as an image and extract text
data = {}
for i, image in enumerate(images):
    image_path = f'output_page_{i + 1}.png'
    image.save(image_path, 'PNG')
    print(f'Saved {image_path}')
    text = extract_text_from_image(image)
    contextualized_text = add_context_to_text(text)
    data[f'page_{i + 1}'] = contextualized_text

# Save the extracted text with additional context to a JSON file
with open('extracted_text_with_context.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)

print('Text extraction and contextualization complete. Data saved to extracted_text_with_context.json')