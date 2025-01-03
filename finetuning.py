import torch
import json
from PIL import Image
from datasets import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from dotenv import load_dotenv

load_dotenv()
model_id = "google/paligemma2-3b-pt-448"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

device = "cuda"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager").to(device) #quantization_config=bnb_config)
#model = get_peft_model(model, lora_config)
#model.print_trainable_parameters()
processor = PaliGemmaProcessor.from_pretrained(model_id)
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")

def collate_fn(examples):
    texts = ["<image>" + example["prefixes"] + "<bos>" for example in examples]
    labels = [example['suffixes'] + "<eos>" for example in examples]
    processed_images = []
    for example in examples:
        image = Image.open(f"{example['images']}").convert("RGB")
        processed_images.append(image)
    tokens = processor(text=texts, images=processed_images, suffix=labels,
                      return_tensors="pt", padding="longest")
    for img in processed_images:
        img.close()
    processed_images = []
    tokens = tokens.to(torch.bfloat16).to(device)
    torch.cuda.empty_cache()
    return tokens

#for param in model.vision_tower.parameters():
#    param.requires_grad = False

#for param in model.multi_modal_projector.parameters():
#    param.requires_grad = False

args = TrainingArguments(
    num_train_epochs=10,
    remove_unused_columns=False,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    warmup_steps=2,
    learning_rate=2e-5,
    weight_decay=1e-6,
    adam_beta2=0.999,
    logging_steps=100,
    optim="adamw_hf",
    save_strategy="steps",
    save_steps=1000,
    push_to_hub=True,
    save_total_limit=1,
    output_dir="paligemma2_thinking_v2",
    bf16=True,
    report_to=["tensorboard"],
    dataloader_pin_memory=False
)

train_ds = Dataset.from_dict(json.load(open("training_data/truncated_training_dict_v2.json")))

trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    data_collator=collate_fn,
    args=args
)

trainer.train()
trainer.push_to_hub()