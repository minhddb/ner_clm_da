from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


PROMPT = "Text: "
model = "clm_ft_Qwen3-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model)
model = AutoModelForCausalLM.from_pretrained(model)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

inputs = tokenizer(PROMPT, return_tensors="pt")
print(inputs)

for _ in range(10):
    output = model.generate(input_ids=inputs["input_ids"], 
                            attention_mask=inputs["attention_mask"], 
                            max_new_tokens=100,
                            do_sample=True, 
                            top_k=50, 
                            top_p=0.9,
                            temperature=0.8)
    augmented_text = tokenizer.batch_decode(output)
    print(augmented_text[0])
    print()