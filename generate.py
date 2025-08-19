import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from utils import LoadBIOToDataset, Mappings, SequenceLinearisation, SequenceGenerator

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = "Qwen/Qwen3-8B"
# Prepare dataset for SFT a Causal Language Model
wnut16, _, _ = LoadBIOToDataset(path_to_train_file="data/wnut16/train.conll",
                                path_to_validation_file="data/wnut16/dev.conll",
                                path_to_test_file="data/wnut16/test.conll"
                                )()
wnut16_train = wnut16["train"]
sequence_generator = SequenceGenerator(wnut16_train, words_column="tokens", tags_column="ner_tags")

# Use sentences with at least one entity mention for SFT
entity_sequences = [seq for seq in sequence_generator.get_entity_sequence()]

# Create mappings
mappings = Mappings(entity_sequences, "ner_tags")
entity_to_dist_mappings = mappings.map_entities_to_sequence_distribution()
stratified_sampling = [] # Use this list if stratified sampling is required -> Later usage
# Linearisation step
lineraised_dict = dict(text=[])
for sequence in entity_sequences:
    lineariser = SequenceLinearisation(sequence["tokens"], sequence["ner_tags"])
    lineraised_dict["text"].append(" ".join(lineariser.span_wise()))
linearised_dataset = Dataset.from_dict(lineraised_dict)

# Add <TAGS> to tokeniser as additional special tokens -> Prevent those tags to be tokenised
# additional_special_tokens = {"additional_special_tokens": list(set([tag for sent in linearised_dataset["text"] for tag in sent.split() if tag.startswith("<")]))}
# Configurate tokenizer and overwrite some special tokens
tokenizer = AutoTokenizer.from_pretrained(base_model)
# tokenizer.add_special_tokens(additional_special_tokens)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto")
model.config.pad_token_id = tokenizer.pad_token_id
model.resize_token_embeddings(len(tokenizer))

clm_pipline = pipeline("text-generation",
                    model=model, 
                    tokenizer=tokenizer, 
                    # device=device, 
                    torch_dtype=torch.float16, 
                    temperature=0.8
                    )

def augment(examples, tag, n):
    prompt = ("You are a text augmentation assistant specialised for Named Entity Recognition (NER). " 
                "Your task is to generate high-quality synthetic sentences. Each sentence must contain entities annotated using open and closing XML-style tags (<person>, </person>). "
                f"Provide the output in a json format, where the key is the required entity and the value a list of strings with {n} sentences. "
                f"For example if the input tag is <person>, your json output should be: {tag}: {examples}" 
    )
    print(prompt)
    print(50 * "=")
    outputs = clm_pipline(prompt, max_new_tokens=500, do_sample=True, temperature=0.8, top_p=0.95, eos_token_id=tokenizer.eos_token_id)
    return outputs[0]["generated_text"].split("\n")



print(augment(examples=["My name is <person> Naveed </person> and I am working for <org> CCL </org>",
                        "This is the phone number of <person> Philipp </person>. Please contact him if needed.",
                        "<org> FAU </org> is an university located in <loc> Erlangen </loc> and <loc> Nuremberg </loc>. There are about 35000 students enrolled for different degrees."
], tag="<person>", n=3))
