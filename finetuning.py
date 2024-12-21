import os
import torch
import json
from PIL import Image
from huggingface_hub import login
from datasets import Dataset
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration, TrainingArguments, Trainer

login(token=os.environ.get("HUGGINGFACE_TOKEN"))
model_id = "google/paligemma2-3b-pt-448"
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = "cuda"
image_token = processor.tokenizer.convert_tokens_to_ids("<image>")
def collate_fn(examples):
  texts = ["<image>" + example["prefixes"] + "<bos>" for example in examples]
  labels= [example['suffixes'] + "<eos>" for example in examples]
  images = [Image.open(f"{example['images']}").convert("RGB") for example in examples]
  tokens = processor(text=texts, images=images, suffix=labels,
                    return_tensors="pt", padding="longest")
  tokens = tokens.to(torch.bfloat16).to(device)
  return tokens

model = PaliGemmaForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)

for param in model.vision_tower.parameters():
    param.requires_grad = False

for param in model.multi_modal_projector.parameters():
    param.requires_grad = False

args = TrainingArguments(
            num_train_epochs=10,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
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
            output_dir="paligemma2_thinking",
            bf16=True,
            report_to=["tensorboard"],
            dataloader_pin_memory=False
        )

train_ds = Dataset.from_dict(json.load(open("training_dict.json")))

trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        data_collator=collate_fn,
        args=args
        )

trainer.train()
trainer.push_to_hub()