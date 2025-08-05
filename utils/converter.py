from datasets import Dataset
from typing import Dict, List

class ConvertToDataset:
    def __init__(self, augmentation: Dict, strategy: str | List[str]=None):
        self.augmentation = augmentation
        self.strategy = strategy

    def __call__(self, *columns):
        dataset_dict = dict()
        for augmented in self.get_augmentation():
            # Make sure that number of input columns and augmented data matches
            assert len(augmented) == len(columns), f"{len(augmented)}, {len(columns)}"
            # We expect the order of augmented entry to be the same as in given columns.
            # Hence, first entry should be a list of tokens followed by further lists of tags
            for i, column in enumerate(columns):
                if column not in dataset_dict:
                    dataset_dict.update({column: [augmented[i]]})
                else:
                    dataset_dict[column].append(augmented[i])
        return Dataset.from_dict(dataset_dict)

    def get_augmentation(self):
        if self.strategy:
            self.strategy = [self.strategy] if str == type(self.strategy) else self.strategy
            for strategy in self.strategy:
                yield from self.augmentation[strategy]
        else:
            for strategy in self.augmentation:
                yield from self.augmentation[strategy]
