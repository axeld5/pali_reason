import torch
import json
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from dotenv import load_dotenv
from datasets import Dataset
from PIL import Image

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
                    "image": sample["images"],
                },
                {
                    "type": "text",
                    "text": sample['prefixes'],
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": sample["suffixes"]
                }
            ],
        },
    ]

train_ds = Dataset.from_dict(json.load(open("training_data/training_dict.json")))
train_dataset = [format_data(sample) for sample in train_ds]

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = Image.open(sample[1]['content'][0]['image'])
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_inputs.append([image])

    # Prepare the inputs for the model
    model_inputs = processor(
        #text=[text_input],
        text=text_input,
        images=image_inputs,
        return_tensors="pt",
    ).to(device)  # Move inputs to the specified device

    # Generate text with the model
    generated_ids = model.generate(**model_inputs, max_new_tokens=max_new_tokens)

    # Trim the generated ids to remove the input ids
    trimmed_generated_ids = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the output text
    output_text = processor.batch_decode(
        trimmed_generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    return output_text[0]

output = generate_text_from_sample(model, processor, train_dataset[1])