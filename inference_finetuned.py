import os
import pandas as pd
from huggingface_hub import login
from transformers import PaliGemmaProcessor, PaliGemmaForConditionalGeneration
from datasets import load_dataset

login(token=os.environ.get("HUGGINGFACE_TOKEN"))
model_id = "google/paligemma2-3b-pt-448"
processor = PaliGemmaProcessor.from_pretrained(model_id)
device = "cuda"

model_id = "axel-darmouni/paligemma2_thinking"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).to(device)

dataset = load_dataset("AI4Math/MathVista")
prompts = []
result_list = []
true_answers = []
for i in range(351, len(dataset["testmini"])):
    prompt = dataset["testmini"][i]["query"]
    raw_image = dataset["testmini"][i]["decoded_image"]
    paligemma_prompt = "<image>" + prompt + "<bos>"
    inputs = processor(paligemma_prompt, raw_image.convert("RGB"), return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=2000)
    result = processor.decode(output[0], skip_special_tokens=True)[len(prompt):]
    prompts.append(prompt)
    result_list.append(result)
    true_answers.append(dataset["testmini"][i]["answer"])
df = pd.DataFrame({"prompt": prompts, "result": result_list, "true_answer": true_answers})
df.to_csv("results_finetuned.csv", index=False)