import torch
import json
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
from datasets import Dataset

load_dotenv()
model_id = "HuggingFaceTB/SmolVLM-Instruct"
model = Idefics3ForConditionalGeneration.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)

processor = AutoProcessor.from_pretrained(model_id)

system_message = """You are a Vision Language Model specialized in handling mathematical questions with reasoning
Your task is, given a question that requires thinking, to give the right answer to it."""

def format_data(sample):
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_message
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": sample["image"],
                },
                {
                    "type": "text",
                    "text": sample['query'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["label"][0]
                }
            ],
        },
    ]

train_ds = Dataset.from_dict(json.load(open("training_data/training_dict.json")))
print(train_ds)