import os

from qwen_vl_utils import process_vision_info
from transformers import (AutoProcessor, AutoTokenizer,
                          Qwen2_5_VLForConditionalGeneration)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Load all images from the ./image_output folder
image_folder = './image_output2'
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(image_files)

# Messages containing multiple images and a text query
messages = [
    {
        "role": "user",
        "content": [{"type": "image", "image": os.path.join(image_folder, img)} for img in image_files] + 
                   [{"type": "text", "text": "Analyse this image and give a array of objects response, here there are details about each person in a voters list, each box denotes a person details like name, age, sex, door no, father's name also sometimes husband name, voterID at top right of each box with 10 digit character, all other these details will be in tamil and id at top left  it will have only numbers in it with 3 or 4 characters. IF data isn't available for specific key use null as value, if data is na then use null as value, extract all the persons data in a array of objects response of all, if text is similar to கணவர் பெயர் means husband's name, if text is similar to தந்தை பெயர் it means father's name, so next time when you generate keep this in mind, if the image doesn't have the relevant information skip that image, don't generate any response, also don't even response with any message just return empty string"}],
    }
]

print(messages)

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=64)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)