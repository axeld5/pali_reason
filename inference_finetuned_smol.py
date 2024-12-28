import torch
import pandas as pd
from transformers import Idefics3ForConditionalGeneration, AutoProcessor
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
model_id = "HuggingFaceTB/SmolVLM-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
thinking_model_id = "axel-darmouni/smolvlm-instruct-thinking"
model = Idefics3ForConditionalGeneration.from_pretrained(
    thinking_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    _attn_implementation="flash_attention_2",
)


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
                    "text": sample['input'],
                }
            ],
        },
    ]

def generate_text_from_sample(model, processor, sample, max_new_tokens=1024, device="cuda"):
    # Prepare the text input by applying the chat template
    text_input = processor.apply_chat_template(
        sample[1:2],  # Use the sample without the system message
        add_generation_prompt=True
    )

    image_inputs = []
    image = sample[1]['content'][0]['image']
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

dataset = load_dataset("AI4Math/MathVista")
prompts = []
result_list = []
true_answers = []
for i in range(351, len(dataset["testmini"])):
    print(f"step {i - 351}")
    prompt = dataset["testmini"][i]["query"]
    raw_image = dataset["testmini"][i]["decoded_image"]
    inputs = format_data(sample = {"image": raw_image, "input": prompt})
    output = generate_text_from_sample(model, processor, inputs)
    if i-351 <= 20:
        print(output)
    prompts.append(prompt)
    result_list.append(output)
    true_answers.append(dataset["testmini"][i]["answer"])
df = pd.DataFrame({"prompt": prompts, "result": result_list, "true_answer": true_answers})
df.to_csv("outputs/results_finetuned_smol.csv", index=False)