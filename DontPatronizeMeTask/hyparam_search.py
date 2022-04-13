import optuna, logging, torch, os, transformers, tqdm, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# import custom modules
import data_loader, data_utils
from data_utils import PatronizeDataset
from train_utils import Trainer_patronize, eval_autoModel, build_torch_dataset_from_df
from backbone_configs import BACKBONE_DICT

BACKBONES = list(BACKBONE_DICT.keys())
print("possible backbones:", BACKBONES)

def load_data(trial, tokenizer):
    # Loading data
    dont_patronize = data_loader.DPT_Dataloader()
    train, val, test = (
        dont_patronize.train_df,
        dont_patronize.val_df,
        dont_patronize.test_df,
    )
    val["text"] = val['text'].str.replace('[^\w\s]','')
    test["text"] = test['text'].str.replace('[^\w\s]','')
    # loading in precomputed data augmentation to oversample pos class
    train_aug_translated = pd.read_csv("./train_translated_aug.csv")
    train_aug_cont_embd = pd.read_csv("./context_word_embed_aug.csv")
    train_aug = pd.concat((train, train_aug_translated, train_aug_cont_embd))
    train_aug["text"] = train_aug['text'].str.replace('[^\w\s]','')
    train_pos = train_aug[train_aug.label >= 0.5]
    print("pos-class ratio:",len(train_pos)/len(train_aug))
    trial.set_user_attr("ratio-pos-class", len(train_pos)/len(train))
    MAX_SEQ_LENGTH = trial.suggest_int("max-seq-length", 16, 256, 16)
    print(train_aug.head(30))

    # Building PyTorch datasets for training
    train_dataset = PatronizeDataset(tokenizer, train_aug, max_length=MAX_SEQ_LENGTH)
    return train_dataset, val, test


def build_model(trial):
    BACKBONE = trial.suggest_categorical("back-bone", BACKBONES)
    tokenizer = transformers.AutoTokenizer.from_pretrained(BACKBONE)
    model = transformers.AutoModelForSequenceClassification.from_pretrained(  # TODO find out how to freeeze layers
    BACKBONE,
    num_labels=1,
    output_attentions=False,
    output_hidden_states=False,
    )
    return model, tokenizer


def train(trial, model, train_dataset: PatronizeDataset):
    training_args = transformers.TrainingArguments(
        output_dir='./experiment/patronize',
        learning_rate = trial.suggest_float("lr", 1e-6, 1e-2),
        logging_steps = 10,
        per_device_train_batch_size=trial.suggest_int("batch-size", 16, 256, 32),
        warmup_steps=trial.suggest_int("lr-warmup-steps", 16, 128, 16),
        num_train_epochs=trial.suggest_int("num-epochs", 2, 20, 1),
        weight_decay=trial.suggest_float("weight-decay", 0.0, 1e-2),
        fp16=True,
    )
    trainer = Trainer_patronize(
            pos_weight=torch.Tensor([trial.suggest_float("bce_weight", 1.0, 5.0)]),
            model=model,                         
            args=training_args,                 
            train_dataset=train_dataset,
            data_collator=train_dataset.collate_fn,
        )
    try:
        trainer.train()
    except Exception as e:
        trial.set_user_attr("Error", str(e))
        print(f"Oh snap, somehting went wrong trying to train {model}, {type(model)}")
        return None
    return trainer

def objective(trial):
    model, tokenizer = build_model(trial)
    train_dataset, val, test = load_data(trial, tokenizer)
    trainer = train(trial, model, train_dataset)
    if trainer is None:
        return 0.
    # gotta rebuild datasets for inference with AutoModel
    val_ds = build_torch_dataset_from_df(val, tokenizer)
    test_ds = build_torch_dataset_from_df(test, tokenizer)
    report = eval_autoModel(dataset=val_ds, model=model, return_dict=False)
    print("Val-set classification report", "\n", report)
    report = eval_autoModel(dataset=val_ds, model=model, return_dict=True)
    f1_pos = report['1']['f1-score']
    # logging test-set performance (not used for model selection!)
    report_test = eval_autoModel(dataset=test_ds, model=model, return_dict=True)  
    f1_pos_test = report_test['1']['f1-score']
    trial.set_user_attr("test-f1-score", f1_pos_test)
    print(f".....  achieved f1-score on pos class of {f1_pos}")
    return f1_pos

def main():
    # Add stream handler of stdout to show the messages
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = f"hyperparam-search#2"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)
    study = optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(),
        )
    study.optimize(objective, n_trials=2000, timeout=None)

if __name__ == "__main__":
    main()