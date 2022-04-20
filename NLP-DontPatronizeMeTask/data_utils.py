import torch

def labels2file(p, outf_path):
	with open(outf_path,'w') as outf:
		for pi in p:
			outf.write(','.join([str(k) for k in pi])+'\n')


class PatronizeDataset(torch.utils.data.Dataset):

    def __init__(self, tokenizer, input_set, max_length=128):

        self.tokenizer = tokenizer
        self.texts = input_set[['text']].values
        self.labels = input_set[['label']].values
        self.max_length=max_length
        
    def collate_fn(self, batch):

        texts = []
        labels = []
        for b in batch:
            texts.append((b['text'][0]))
            labels.append(b['label'][0])   
        encodings = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=self.max_length)
        encodings['label'] = torch.tensor(labels)
        
        return encodings
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
       
        item = {'text': self.texts[idx],
                'label': self.labels[idx]
                }
        return item