import pandas as pd
from urllib import request
from sklearn.model_selection import train_test_split

SEED = 42

from dont_patronize_me import DontPatronizeMe


class DPT_Dataloader:
    def __init__(self, val_split: float = 0.2):
        self.dpm = DontPatronizeMe(".", ".")
        train_df, test_df = self.load_data()
        train_df, val_df = train_test_split(
            train_df, test_size=val_split, random_state=SEED
        )
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

    def load_data(self):
        self.dpm.load_task1()
        self.dpm.load_task2(return_one_hot=True)
        train_ids, test_ids = self.load_paragraph_ids()
        train_df = self.build_dataset_from_ids(train_ids)
        test_df = self.build_dataset_from_ids(test_ids)
        return train_df, test_df

    def load_paragraph_ids(self):
        module_dev_csv_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/practice%20splits/dev_semeval_parids-labels.csv"
        module_train_csv_url = f"https://raw.githubusercontent.com/Perez-AlmendrosC/dontpatronizeme/master/semeval-2022/practice%20splits/train_semeval_parids-labels.csv"

        module_dev_csv_name = module_dev_csv_url.split("/")[-1]
        module_train_csv_name = module_train_csv_url.split("/")[-1]
        print(f"Fetching {module_dev_csv_name}")
        print(f"Fetching {module_train_csv_name}")

        train_ids = pd.read_csv(module_train_csv_url)
        test_ids = pd.read_csv(module_dev_csv_url)
        train_ids.par_id = train_ids.par_id.astype(str)
        test_ids.par_id = test_ids.par_id.astype(str)
        return train_ids, test_ids

    def build_dataset_from_ids(self, id_df: pd.DataFrame, train_set: bool = True):
        rows = []  # will contain par_id, label and text
        data_df = self.dpm.train_task1_df if train_set else self.dpm.test_task1_df
        for idx in range(len(id_df)):
            parid = id_df.par_id[idx]
            text = data_df.loc[data_df.par_id == parid].text.values[0]
            label = data_df.loc[data_df.par_id == parid].label.values[0]
            rows.append({"par_id": parid, "text": text, "label": label})
        return pd.DataFrame(rows)
