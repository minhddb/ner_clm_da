from typing import List, Dict
from datasets import Dataset
from dataset import Data, DatasetLoader, SequenceSegmentation
from dataset.utils import EntityCount, ContextEntityExtraction

from itertools import chain

class Linearisation:
    def __init__(self, dataset: Dataset, ids_to_labels_mapping: Dict, tag_name: str = "ner_tags"):
        self.dataset = dataset
        self.ids_to_labels_mapping = ids_to_labels_mapping
        self.tag_name = tag_name
    pass


class SequenceLinearisation: 
    def __init__(self, sequence: List[str], tags: List[str]):
        self.BOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"
        self.sequence = sequence
        self.tags = tags
        self.extractor = ContextEntityExtraction(self.sequence, self.tags)
        self.entity_tokens_ids = list(
            chain.from_iterable(
                self.extractor.extract_entity()[2]
            )
        )
        self.context_tokens_ids = list(chain.from_iterable(
            self.extractor.extract_context_windows_ids(windows_size=1)
        )
        )

    def __call__(self, mode="span"):
        assert mode in ["span", "label"]
        if mode == "span":
           return " ".join(self.span_wise())
        if mode == "label":
            return " ".join(self.label_wise())
   
    def span_wise(self):
        """
        Insert entity tags before and after each entity span within the sequence.
        :return: List of linearised tokens.
        E.g.: [Token_1, Token_2, <TAG1>, Token_3, Token_4, </TAG1>, <TAG2>, Token_5 </TAG2>, Token_6, Token_7]
        """
        linearised = [self.BOS_TOKEN]
        for i, token in enumerate(self.sequence):
            if i not in self.entity_tokens_ids:
                linearised.append(token)
            else:
                entity = tags[i].strip("B-").strip("I-")
                if tags[i].startswith("B-"):
                    linearised.append(f"<{entity}>")
                    linearised.append(token)
                elif tags[i + 1].strip("B-").strip("I-") != entity:
                    linearised.append(token)
                    linearised.append(f"</{entity}>")
                else:
                    linearised.append(token)
        assert all(token in linearised for token in self.sequence)
        linearised.append(self.EOS_TOKEN)
        return linearised
    
    def label_wise(self):
        """
        """
        linearised = []
        for i, token in enumerate(self.sequence):
            if i not in self.entity_tokens_ids:
                linearised.append(token)
            else:
                linearised.append(f"<{tags[i]}>")
                linearised.append(token)
                linearised.append(f"</{tags[i]}>")
        return linearised

    @staticmethod
    def context_tag():
        pass

    @staticmethod
    def entity_tag():
        pass


if __name__=="__main__":
    tokens = ['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice',
              'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad',
              'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']
    tags = ['O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    linearisation = SequenceLinearisation(tokens, tags)
    print(linearisation())
    print(linearisation(mode="label"))


    # entity_counter = EntityCount(train_dataset, mapping_dict=ids2labels)
    # entity_distribution = entity_counter.count_occurrences()
    print()
