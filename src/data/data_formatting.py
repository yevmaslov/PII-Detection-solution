import ast
import string
from spacy.lang.en import English
en_tokenizer = English().tokenizer

def tokenize_with_spacy(text, tokenizer=en_tokenizer):
    tokenized_text = tokenizer(text)
    tokens = [token.text for token in tokenized_text]
    trailing_whitespace = [bool(token.whitespace_) for token in tokenized_text]
    return {'tokens': tokens, 'trailing_whitespace': trailing_whitespace}

def pj_to_pj(full_text, student_data):
    labels_ = {key: [value] for key, value in student_data.items() if value}
    # row = {'full_text': full_text, 'labeldict': ners}
    labeldict={}
    # labels_ = ast.literal_eval(row["labeldict"])
    
    for k, v in labels_.items():
        v_processed=[]
        
        for x in v:
            x = x.split(" ")
            for n in x:
                while n[-1] in string.punctuation: 
                    n=n[:-1]
                v_processed+=[n]
            
            if k == "PHONE_NUM":
                for x in v:
                    x=x.split("-")
                    for n in x:
                        while n[-1] in string.punctuation: 
                            n=n[:-1]
                        v_processed+=[n]
            if k == "ID_NUM":
                for x in v:
                    x=x.split("-")
                    for n in x:
                        while n[-1] in string.punctuation: 
                            n=n[:-1]
                        v_processed+=[n]
        labeldict[k] = v_processed
    return labeldict


def interpolate_labels(sample):
    punctuation = [p for p in string.punctuation]
    sandwich_on_comma = ["STREET_ADDRESS"]
    t = sample["tokens"]
    ws = sample["trailing_whitespace"]
    label = sample["labels"]
    full_text = sample['full_text']
    new_labels=["O"]*len(label)
    
    for i, l in enumerate(label):
        if i != 0: prior_label=label[i-1]
        else: prior_label="O"
            
        if i+1 < len(label): next_label=label[i+1]
        elif i+1 == len(label): next_label="O"
            
        if (t[i] == "and" and l == "O") or (t[i] == "or" and l == "O"):
            new_labels[i] = "O"
            
        elif prior_label == "EMAIL" and t[i] == "to":
            new_labels[i] = "O"
    
        elif t[i] == "," and prior_label not in sandwich_on_comma:
            new_labels[i] = "O"
            
        elif prior_label == next_label and prior_label != "O":
            new_labels[i] = prior_label
        elif l != "O":
            new_labels[i] = l
        else:
            new_labels[i] = "O"
    
    sample = {"tokens": t, "trailing_whitespace": ws, "labels":new_labels, 'full_text': full_text}
    return sample

def convert_to_bio(sample):
    t = sample["tokens"]
    ws = sample["trailing_whitespace"]
    label = sample["labels"]
    full_text = sample['full_text']
    
    last_label="O"
    for i, l in enumerate(label):
        if l != last_label and l != "O":
            label[i] = "B-"+l
        elif l == last_label and last_label != "O":
            label[i] = "I-"+l
        last_label = l
    sample = {"tokens": t, "trailing_whitespace": ws, "labels": label, 'full_text': full_text}
    return sample

def format_output(full_text, student_data):
    tokens = tokenize_with_spacy(full_text)
    labels = pj_to_pj(full_text, student_data)

    t = tokens["tokens"]
    ws = tokens["trailing_whitespace"]
    new_labels=["O"]*len(t)
    for ent_type, ent_list in labels.items():
        for ent_ in ent_list:
            indices = [i for i, x in enumerate(t) if x == ent_]
            for i in indices:
                new_labels[i] = ent_type
    sample = {"tokens": t, "trailing_whitespace": ws, "labels": new_labels, 'full_text': full_text}

    sample = interpolate_labels(sample)
    sample = convert_to_bio(sample)
    return sample
