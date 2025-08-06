from typing import List, Dict
from utils.segmentation import SequenceSegmentation 
from itertools import chain
import re


class SequenceLinearisation(SequenceSegmentation): 
    def __init__(self, sequence: List[str], tags: List[str], domain_tag=None):
        super().__init__(sequence=sequence, tags=tags)
        self.BOS_TOKEN = "<s>"
        self.EOS_TOKEN = "</s>"
        if domain_tag is not None:
            self.domain_tag = domain_tag
        self.entity_tokens_ids = list(
            chain.from_iterable(
                [tokens_ids for tokens_ids in self.get_entity_tokens_ids()]
            )
        )
        self.context_tokens_ids = list(chain.from_iterable(
            self.get_context_windows_tokens_ids(windows_size=1)
        )
        )

    def __call__(self, mode="span"):
        assert mode in ["span", "label"]
        if mode == "span":
           return " ".join(self.span_wise())
        if mode == "label":
            return " ".join(self.label_wise())
    
    def context_wise(self):
        """
        """
        pass

    def span_wise(self):
        """
        Insert entity tags before and after each entity span within the sequence.
        :return: List of linearised tokens.
        E.g.: [Token_1, Token_2, <TAG1>, Token_3, Token_4, </TAG1>, <TAG2>, Token_5 </TAG2>, Token_6, Token_7]
        """
        linearised = [] # [self.BOS_TOKEN]
        for i, token in enumerate(self.sequence):
            if i in self.entity_tokens_ids:
                entity = re.sub(r"^[BI]-", "", self.tags[i])
                if self.tags[i].startswith("B-"):
                    linearised.append(f"<{entity}>")
                    linearised.append(token)
                    if len(self.tags) <= i+1 or re.sub(r"^[BI]-", "", self.tags[i+1]) != entity:
                        linearised.append(f"</{entity}>")
                elif len(self.tags) <= i+1 or re.sub(r"^[BI]-", "", self.tags[i+1]) != entity:
                    linearised.append(token)
                    linearised.append(f"</{entity}>")      
            else:
                linearised.append(token)
        # linearised.append(self.EOS_TOKEN)
        return linearised
    
    def label_wise(self):
        """
        """
        linearised = [self.BOS_TOKEN]
        for i, token in enumerate(self.sequence):
            if i not in self.entity_tokens_ids:
                linearised.append(token)
            else:
                linearised.append(f"<{self.tags[i]}>")
                linearised.append(token)
                linearised.append(f"</{self.tags[i]}>")
        linearised.append(self.EOS_TOKEN)
        return linearised


if __name__=="__main__":
    tokens = ['The', 'European', 'Commission', 'said', 'on', 'Thursday', 'it', 'disagreed', 'with', 'German', 'advice',
              'to', 'consumers', 'to', 'shun', 'British', 'lamb', 'until', 'scientists', 'determine', 'whether', 'mad',
              'cow', 'disease', 'can', 'be', 'transmitted', 'to', 'sheep', '.']
    tags = ['O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O', 'O', 'O', 'O', 'B-MISC', 'O', 'O',
            'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']
    linearisation = SequenceLinearisation(tokens, tags)
    print(" ".join(tokens))
    print(linearisation())
    print(linearisation(mode="label"))
    print()
