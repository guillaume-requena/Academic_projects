import logging, torch, os, transformers, tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# import custom modules
import data_loader, data_utils
from data_utils import PatronizeDataset
from train_utils import Trainer_patronize, compute_metrics, evaluate

# logging
logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)

TEST_MODE=True # SET TO TRUE for CODALAB predictions

# Hyperparams
BACKBONE = "bert-base-cased"
BATCH_SIZE = 176
NUM_EPOCHS = 3
LR = 0.00006
LR_WARMUP_STEPS = 80
WEIGHT_DECAY = 0.009
MAX_SEQ_LENGTH = 160
BCE_WEIGHT = 4.09

# Loading data
dont_patronize = data_loader.DPT_Dataloader()
train, val, test = (
    dont_patronize.train_df,
    dont_patronize.val_df,
    dont_patronize.test_df,
)
val["text"] = val["text"].str.replace("[^\w\s]", "")
test["text"] = test["text"].str.replace("[^\w\s]", "")

# loading in pre-computed augmented training data
train_aug_translated = pd.read_csv("./train_translated_aug.csv")
train_aug_cont_embd = pd.read_csv("./context_word_embed_aug.csv")

if TEST_MODE:
    train = pd.concat((train, val))

#downsampling and applying data augmentation
train_aug = pd.concat((train, train_aug_translated, train_aug_cont_embd))
train_aug["text"] = train_aug["text"].str.replace("[^\w\s]", "")
train_pos = train_aug[train_aug.label >= 0.5]
print("pos-class ratio before downsampling:", len(train_pos) / len(train_aug))
npos = len(train_pos)
print("pos-class ratio after downsampling:", len(train_pos) / len(train_aug))
print(train_aug.head())

# Instantiating tokenizer and model
tokenizer = transformers.AutoTokenizer.from_pretrained(BACKBONE)
model = transformers.AutoModelForSequenceClassification.from_pretrained(
    BACKBONE,
    num_labels=1,
    output_attentions=False,
    output_hidden_states=False,
)
# Building PyTorch datasets for training and testing
train_dataset = PatronizeDataset(tokenizer, train_aug, max_length=MAX_SEQ_LENGTH)
val_dataset = PatronizeDataset(tokenizer, val, max_length=MAX_SEQ_LENGTH)
test_dataset = PatronizeDataset(tokenizer, test, max_length=MAX_SEQ_LENGTH)

# main training logic
training_args = transformers.TrainingArguments(
    output_dir="./experiment/patronize",
    learning_rate=LR,
    logging_steps=10,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_steps=LR_WARMUP_STEPS,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    fp16=True,
)
trainer = Trainer_patronize(
    pos_weight=torch.Tensor([BCE_WEIGHT]),
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=train_dataset.collate_fn,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(f"./models/{BACKBONE}_finetuned/")

# evaluation
# annoying but we have to reformat our dataset for predicting with AutoModels...
from train_utils import eval_autoModel, build_torch_dataset_from_df, predict_autoModel
test_ds = build_torch_dataset_from_df(test, tokenizer)

if not TEST_MODE:
    val_ds = build_torch_dataset_from_df(val, tokenizer)
    report = eval_autoModel(dataset=val_ds, model=model, return_dict=False)
    print("Val-set classification report", "\n", report)

report = eval_autoModel(dataset=test_ds, model=model, return_dict=False)
print("Test-set classification report", "\n", report)

#predicting for codalab
from data_utils import labels2file
test_codalab = pd.read_csv("task4_test.tsv", sep='\t', names=["par_id", "address", "category", "country", "text"] , header=None)
test_codalab_text = test_codalab.drop(columns=["address", "category", "country"])
test_coda_ds = build_torch_dataset_from_df(test_codalab_text, tokenizer, predict_only=True)
preds = predict_autoModel(test_coda_ds, model)
print(preds)
labels2file([[k] for k in preds], 'task1.txt')


