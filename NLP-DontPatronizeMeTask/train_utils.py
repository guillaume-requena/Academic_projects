import transformers, torch, tqdm
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from data_utils import PatronizeDataset
import numpy as np
from tqdm import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torch.utils.data import TensorDataset


class Trainer_patronize(transformers.Trainer):
    
    def __init__(self, pos_weight=None, **kwargs):
        super().__init__(**kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs):
        
        labels = inputs.pop('label').to(device)
        predictions = model(**inputs, return_dict=False)
        predictions = predictions[0] if isinstance(predictions, tuple) else predictions
        predictions = predictions.to(device)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight).to(device=device)
        loss = bce(predictions.float().view(-1), labels.float().view(-1))

        return loss

def compute_metrics(p:transformers.EvalPrediction):    
    pred, labels = p
    pred = torch.nn.Sigmoid()(pred)
    pred = np.where(pred>=0.5,1, 0)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    recall = recall_score(y_true=labels, y_pred=pred)
    precision = precision_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred)    
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} 

def evaluate(trainer:transformers.Trainer, dataset:PatronizeDataset, return_dict:bool=False):
    preds = torch.Tensor(trainer.predict(dataset).predictions)
    preds = torch.nn.Sigmoid()(preds)
    preds = np.where(preds.numpy() >= 0.5, 1, 0)
    labels = np.where(dataset.labels >= 0.5, 1, 0)
    report = classification_report(y_true=labels, y_pred=preds, output_dict=return_dict)
    return report

def eval_autoModel(dataset, model, return_dict): 
    dat_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    preds, labels = [], []
    model.eval()
    for data in tqdm(dat_loader):
        b_input_ids = data[0].to(device)
        b_input_mask = data[1].to(device)
        y = data[2].to(device)
        logits = model(b_input_ids, b_input_mask).logits.view(-1)
        y_pred = torch.nn.Sigmoid()(logits)
        preds.append(y_pred.cpu().detach().numpy())
        labels.append(y.cpu().detach().numpy())
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    print(preds)
    print(labels)
    preds = np.where(preds >= 0.5, 1, 0)
    labels = np.where(labels >= 0.5, 1, 0)
    report = classification_report(y_true=labels, y_pred=preds, output_dict=return_dict)

    return report

def predict_autoModel(dataset, model):
    dat_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    preds, labels = [], []
    model.eval()
    for data in tqdm(dat_loader):
        b_input_ids = data[0].to(device)
        b_input_mask = data[1].to(device)
        logits = model(b_input_ids, b_input_mask).logits.view(-1)
        y_pred = torch.nn.Sigmoid()(logits)
        preds.append(y_pred.cpu().detach().numpy())
    preds = np.concatenate(preds)
    preds = np.where(preds >= 0.5, 1, 0)
    return preds


def build_torch_dataset_from_df(df, tokenizer, predict_only=False):
        input_ids = []
        attention_masks = []
        for sentence in df.text.values:
            encoded_dict = tokenizer.encode_plus(
                sentence,  # Sentence to encode.
                add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                max_length=128,  # Pad & truncate all sentences.
                pad_to_max_length=True,
                return_attention_mask=True,  # Construct attn. masks.
                return_tensors="pt",  # Return pytorch tensors.
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)
        labels = torch.ones(input_ids.shape[0]) if predict_only else torch.tensor(df.label.values, dtype=torch.float32)
        return TensorDataset(input_ids, attention_masks, labels)
