from utils import DatasetLoader, LoadBIOToDataset, Mappings, SequenceLinearisation, SequenceGenerator
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling,  Trainer, TrainingArguments
from datasets import Dataset

def main():
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
    custom_special_tokens = {"pad_token": "<pad>", "additional_special_tokens": list(set([tag for sent in linearised_dataset["text"] for tag in sent.split() if tag.startswith("<")]))}
    print(custom_special_tokens)

    # Configurate tokenizer and overwrite some special tokens
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
    tokenizer.add_special_tokens(custom_special_tokens)
    tokenizer.padding_size = "left"
    tokenizer.pad_token = "<pad>"

    def tokenize(examples):
        tokenized_output = tokenizer(examples["text"], truncation=True, padding=True, return_tensors="pt")
        # tokenized_output["labels"] = [ids.copy() for ids in tokenized_output["input_ids"]]
        return tokenized_output 
    

    tokenized_dataset = linearised_dataset.map(tokenize, batched=True, num_proc=4, remove_columns=["text"])
    lm_dataset = tokenized_dataset# .map(group_texts, batched=True, num_proc=4)

    print(lm_dataset)
    print(lm_dataset[21])
    print(tokenizer.convert_ids_to_tokens(lm_dataset[21]["input_ids"]))

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
     
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(output_dir="./clm_ft_Qwen3-0.6B", 
                                       eval_strategy="no", 
                                       num_train_epochs=5,
                                       per_device_train_batch_size=8,
                                       gradient_accumulation_steps=2, 
                                       fp16=True,
                                       save_total_limit=1,
                                       learning_rate=1e-5, 
                                       weight_decay=1e-2,
                                       overwrite_output_dir=True,
                                       push_to_hub=False)
    trainer = Trainer(model=model, 
                     args=training_args,
                     train_dataset=lm_dataset, 
                     data_collator=data_collator, 
                     tokenizer=tokenizer
                     )
    trainer.train()
    trainer.save_model()
    # tokenizer.save_pretrained("./clm_ft_Qwen3-0.6B")

if __name__=="__main__":
    main()