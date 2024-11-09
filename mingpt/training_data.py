# create data loader and data set for training

DATA_PATH = '/nobackup/archive/usr/dw87/pile_data_10.jsonl'


# import pytorch dataset
from torch.utils.data import Dataset
import jsonlines

# import huggingface gpt2 tokenizer
from transformers import GPT2Tokenizer


class NaiveJsonlDataset(Dataset):
    """Dataset for jsonl files with "text" field in each line. 
    
    NOTE: This is a naive implementation that loads the entire dataset into memory.
    """
    
    def __init__(self, data_path=DATA_PATH):
        self.data = [line["text"] for line in jsonlines.open(data_path)]
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.max_length = 2048
    
    def __len__(self):
        return len(self.data)
    
    
    def __getitem__(self, idx):
        text = self.data[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded['input_ids'].squeeze(0)  # Tensor of token ids

        # Create labels by shifting input_ids to the right
        labels = input_ids.clone()
        

        return input_ids, labels
