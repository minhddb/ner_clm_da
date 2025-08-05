from datasets import Dataset

class SequenceGenerator:
    def __init__(self,dataset: Dataset, words_column: str, *tags_columns: str):
        self.dataset = dataset
        self.words_column = words_column
        self.tags_columns = tags_columns

    def get_entity_sequence(self):
        """
        Yield sequences with annotated entities only
        """
        for _, sequence in enumerate(self.dataset):
            if any("B-" in label for label in sequence[self.tags_columns[0]]):
                yield sequence
    
    def get_sequence_by_category(self, tag_column, category):
        for _, sequence in enumerate(self.dataset):
            if any(category in label for label in sequence[tag_column]):
                yield sequence

    def yield_data(self):
        yield from self.dataset