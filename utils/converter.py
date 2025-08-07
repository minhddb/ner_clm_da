from datasets import Dataset
from typing import  List
import re

#TODO: Work on an approach to remove entity tags from tokens sequence and create a list of NER labels for training


class ConvertAugmentationToBIO:
    def __init__(self,  aug_sequence: List[str], mode="span"):
        self.aug_sequence = aug_sequence
        self.mode = mode

    def __call__(self, mode="span"):
        assert mode in ["span", "labels"], "Allowd mode: 'span', 'labels'"
        tokens = [token for token in self.aug_sequence if token != "<s>" and not (token.startswith("<") and token.endswith(">"))]
        if mode == "span":
            ner_tags = self.extract_spans()
        if mode == "labels":
            ner_tags = self.extract_labels()
        assert len(tokens) == len(ner_tags), f"{len(tokens)}, {len(ner_tags)}"
        return tokens, ner_tags

    def _yield_from_augmentation(self):
        yield from self.aug_sequence

    def extract_spans(self):
        """
        Extract BIO tags from linearised spans. 
        """
        ner_tags = []
        is_span = False
        entity = None

        for _, token in enumerate(self.aug_sequence):
            if token != "<s>":
                if re.fullmatch(r"<\w+>", token):
                    is_span = True
                    entity = token.strip("<>")
                    continue
                if re.fullmatch(r"<\/\w+>", token):
                    is_span = False
                    continue
                if is_span:
                    if ner_tags[-1] == "O" or ner_tags == []:
                        ner_tags.append(f"B-{entity}")
                    else:
                        ner_tags.append(f"I-{entity}")
                else:
                    ner_tags.append("O")
        return ner_tags

    def extract_labels(self):
        """
        Extract BIO tags from linearised tokens.
        """
        ner_tags = []
        is_label = False
        label = None
        for _, token in enumerate(self.aug_sequence):
            if token != "<s>":
                if token.startswith("<B-") or token.startswith("<I-"):
                    is_label = True
                    label = token.strip("<>")
                elif token.startswith("</B-") or token.startswith("</I"):
                    is_label=False
                else:
                    if is_label:
                        ner_tags.append(label)
                    else:
                        ner_tags.append("O")
        return ner_tags

if __name__ == "__main__":
    linearised_span = "<s> The <ORG> European Commission </ORG> said on Thursday it disagreed with <MISC> German </MISC> advice to consumers to shun <MISC> British </MISC> lamb until scientists determine whether mad cow disease can be transmitted to sheep . "
    linearised_label = "<s> The <B-ORG> European </B-ORG> <I-ORG> Commission </I-ORG> said on Thursday it disagreed with <B-MISC> German </B-MISC> advice to consumers to shun <B-MISC> British </B-MISC> lamb until scientists determine whether mad cow disease can be transmitted to sheep ."
    
    converter = ConvertAugmentationToBIO(linearised_label.split())
    tokens, ner_tags = converter(mode="labels")
    print(tokens, ner_tags, sep="\n")