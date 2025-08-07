from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import torch

base_model = "Qwen/Qwen3-1.7B-Base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = f"./ft_clm_{base_model.split("/")[1]}"
tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)
model.pad_token_id = tokenizer.eos_token_id

ner_tags = [tag for tag in tokenizer.additional_special_tokens if tag != "<s>" and not tag.startswith("</")]
n = 5
for _ in range(n):
    tag = random.sample(ner_tags, 1)
    PROMPT = f"Generate {n} synthetic sentences for NER that have to include {tag[0]}: "
    inputs = tokenizer(PROMPT, return_tensors="pt")
    model.to(device)
    inputs.to(device)
    output = model.generate(input_ids=inputs["input_ids"], 
                                attention_mask=inputs["attention_mask"], 
                                max_new_tokens=50,
                                do_sample=True, 
                                top_p=0.95,
                                temperature=0.8)
    augmented_text = tokenizer.batch_decode(output)
    print(augmented_text[0])
    print()