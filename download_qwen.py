# filepath: d:\Workspace\GenAI\deepseek-rag\download_qwen_model.py
from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

model_name = "Qwen/Qwen2.5-VL-7B-Instruct"
model_path = "./models/qwen_model"

# Download and save the model
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
model.save_pretrained(model_path)

# Download and save the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_path)