import pandas as pd
import numpy as np
import transformers
import nlpaug
import nlpaug.augmenter.word as naw
import data_loader
from tqdm import tqdm

def augment_dataset(df, aug_func):
    training_set_pat = df[df.label != 0.0]
    text = training_set_pat[['text']].values
    labels = training_set_pat[['label']].values
    number_of_samples = len(text)
    augmented_data = []
    for sample in tqdm(range(number_of_samples)):
        new_text = aug_func.augment(text[sample][0], n=1)
        augmented_data.append([8888, new_text, labels[sample][0]])
    augmented_data = np.array(augmented_data)
    df_augmented = pd.DataFrame(augmented_data, columns = ['par_id', 'text', 'label'])
    return df_augmented

# loading DPT data
dont_patronize = data_loader.DPT_Dataloader()
train, val, test = dont_patronize.train_df, dont_patronize.val_df, dont_patronize.test_df
print(train.head())

# using back translation 
back_translation_aug = naw.BackTranslationAug(
    from_model_name='facebook/wmt19-en-de', 
    to_model_name='facebook/wmt19-de-en'
)

augmented_train_translated = augment_dataset(train.copy(), aug_func=back_translation_aug)
print(augmented_train_translated.head())
augmented_train_translated.to_csv("train_translated_aug.csv")

# using contextual word embedding
cont_aug = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased', action="insert")

augmented_context_train = augment_dataset(train.copy(), aug_func=cont_aug)
augmented_context_train.to_csv("context_word_embed_aug.csv")
