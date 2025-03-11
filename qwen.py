import json
import os

import PIL.Image
import torch
from dotenv import load_dotenv
from pydantic import BaseModel
from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)

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

model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Directory containing the images
image_dir = './image_output'

# List to store all responses
all_data_list = []

# Prepare messages for inference
messages = []
for image_file in os.listdir(image_dir):
    if image_file.endswith('.png'):
        image_path = os.path.join(image_dir, image_file)
        messages.append({"type": "image", "image": f"file://{image_path}"})

# Add the context text
context = (
    "Analyse this image and give an array of objects response. "
    "Here there are details about each person in a voters list. "
    "Each box denotes a person's details like name, age, sex, door no, father's name, "
    "sometimes husband's name, voterID at the top right of each box with a 10-digit character. "
    "All other details will be in Tamil and the ID at the top left will have only numbers with 3 or 4 characters. "
    "If data isn't available for a specific key, use null as the value. "
    "If data is not available, use null as the value. "
    "Extract all the person's data in an array of objects response. "
    "If text is similar to கணவர் பெயர், it means husband's name. "
    "If text is similar to தந்தை பெயர், it means father's name. "
    "If the image doesn't have the relevant information, skip that image. "
    "Don't generate any response, and don't even respond with any message, just return an empty string."
)
messages.append({"type": "text", "text": context})

# Preparation for inference
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Process the output text
for response_text in output_text:
    cleaned_response_text = response_text.replace("```json", "").replace("```", "").strip()
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
with open('qwen.json', 'w', encoding='utf-8') as json_file:
    json.dump(all_data, json_file, indent=4, ensure_ascii=False)

print('All responses written to qwen.json')