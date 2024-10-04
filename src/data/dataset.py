from datasets import Dataset
from .utils import filter_no_pii, tokenize
from torch.utils.data import Dataset
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, data_json, max_length, tokenizer, label2id):
        self.data = data_json
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.label2id = label2id
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.data[item]
        
        ml = self.cfg.dataset.max_length
        
        if hasattr(self.cfg.dataset, 'time_shift') and self.cfg.dataset.time_shift:
            out = self.cfg.tokenizer.encode_plus(text, add_special_tokens=False)['input_ids']
            offset = np.random.randint(0, max(1, len(out) - ml - 1))
            out = out[offset:offset+ml]
            text = self.cfg.tokenizer.decode(out)
        
        inputs = self.cfg.tokenizer.encode_plus(
            text,
            return_tensors=None,
            add_special_tokens=True,
            max_length=ml,
            pad_to_max_length=True,
            truncation=True,
        )
        
        if self.train:
            inputs['labels'] = self.labels[item]
            
        return inputs


def get_dataset(data_json, config, tokenizer, label2id):
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data_json],
        "document": [i for i in range(len(data_json))],
        "tokens": [x["tokens"] for x in data_json],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data_json],
        "provided_labels": [x["labels"] for x in data_json],
        "fold": [x["fold"] for x in data_json],
    })
    ds = ds.filter(filter_no_pii, num_proc=2)
    ds = ds.map(
        tokenize, 
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": config.dataset.max_length}, 
        num_proc=2,
    )
    return ds
