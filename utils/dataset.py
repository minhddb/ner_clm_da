import itertools
from typing import Dict, List, Iterable
from dataclasses import dataclass

import numpy as np
from datasets import Dataset, DatasetDict
from datasets import load_dataset


@dataclass
class LoadBIOToDataset:
    path_to_train_file: str = None
    path_to_validation_file: str = None
    path_to_test_file: str = None

    def __call__(self):
        _dataset = dict(train={}, validation={}, test={})
        if self.path_to_train_file is not None:
            train_split = self.create_split(path_to_file=self.path_to_train_file)
        else:
            train_split = {}
        if self.path_to_validation_file is not None:
            validation_split = self.create_split(path_to_file=self.path_to_validation_file)
        else:
            validation_split = {}
        if self.path_to_test_file is not None:
            test_split = self.create_split(path_to_file=self.path_to_test_file)
        else:
            test_split = {}
        _dataset.update({"train": train_split, "validation": validation_split, "test": test_split})
        dataset =  DatasetDict({
            data_split: Dataset.from_dict(data) for data_split, data in _dataset.items()
        }
        )
        labels = sorted(list(set(list((itertools.chain.from_iterable(dataset["train"]["ner_tags"]))))))
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

    def create_split(self, path_to_file):
        tokens, ner_tags = [], []
        data_dict = dict(tokens=[], ner_tags=[])
        for toks, tags in self.sequence_generator(path_to_file):
            tokens.append(toks)
            ner_tags.append(tags)
        data_dict["tokens"] = tokens
        data_dict["ner_tags"] = ner_tags
        return data_dict

    @staticmethod
    def sequence_generator(file_path):
        tokens, tags = [], []
        with open(file_path, "r", encoding="utf-8") as infile:
            lines = infile.readlines()
            for line in lines:
                line = line.strip()
                if line == "":
                    yield tokens, tags
                    tokens, tags = [], []
                else:
                    # Handle weird chracter from original file
                    if line.split()[0] == "":
                        tokens.append("")
                    else:
                        tokens.append(line.split()[0])
                    tags.append(line.split()[1])

@dataclass
class DatasetLoader:
    """
    """
    hf_dataset_name_or_path: str = None

    def __call__(self, *args):
        dataset = load_dataset(self.hf_dataset_name_or_path,*args)
        try:
            label2id = {label: i for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        except KeyError:
            label2id = {label: i for i, label in enumerate(dataset["train"].features["tags"].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

@dataclass
class WNUTDataLoader:
    """
    Load wnut17 dataset from HF
    """
    hf_dataset_name_or_path: str = "leondz/wnut_17"

    def __call__(self):
        dataset = load_dataset(self.hf_dataset_name_or_path, trust_remote_code=True)
        label2id = {label: i for i, label in enumerate(dataset["train"].features["ner_tags"].feature.names)}
        id2label = {i: label for label, i in label2id.items()}
        return dataset, label2id, id2label

class Data:
    def __init__(self, dataset: Dataset, tag_column: str):
        self.dataset = dataset
        self.tag_column = tag_column
        self.non_entity_label = "O"

    def sequence_generator(self):
        for sequence in self._yield_from(self.dataset):
            yield sequence["tokens"], sequence["ner_tags"]

    def get_entity_sequences(self):
        """
        Return all sequences containing at least one entity as list.
        :param ner_tags: Name of entity column.
        """
        entity_sequences = []
        for sequence in self._yield_from(self.dataset):
            if not all(tag == self.non_entity_label for tag in sequence[self.tag_column]):
                ent_sequence = [sequence["tokens"], [tag for tag in sequence[self.tag_column]]]
                if ent_sequence not in entity_sequences:
                    entity_sequences.append(ent_sequence)
        return entity_sequences

    @staticmethod
    def _yield_from(iterable: Iterable):
        yield from iterable
