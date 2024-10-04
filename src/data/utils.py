import random
import numpy as np

def filter_no_pii(example, percent_allow=0.2):
    has_pii = set("O") != set(example["provided_labels"])
    return has_pii or (random.random() < percent_allow)

def tokenize(example, tokenizer, label2id, max_length):
    text = []
    labels = []
    token_map = []
    
    idx = 0
    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):
        text.append(t)
        labels.extend([l]*len(t))
        token_map.extend([idx]*len(t))
        if ws:
            text.append(" ")
            labels.append("O")
            token_map.append(-1)
        
        idx += 1
    
    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length, truncation=True)
    
    labels = np.array(labels)
    
    text = "".join(text)
    token_labels = []
    
    for start_idx, end_idx in tokenized.offset_mapping:
        if start_idx == 0 and end_idx == 0: 
            token_labels.append(label2id["O"])
            continue
        
        if text[start_idx].isspace():
            start_idx += 1
        
        while start_idx >= len(labels):
            start_idx -= 1
            
        token_labels.append(label2id[labels[start_idx]])
        
    length = len(tokenized.input_ids)
        
    return {
        **tokenized,
        "labels": token_labels,
        "length": length,
        "token_map": token_map,
    }
