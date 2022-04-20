import transformers

BACKBONE_DICT = {
    "bert-base-multilingual-cased": transformers.BertModel,
    "bert-base-cased": transformers.BertModel,
    #"openai-gpt": transformers.OpenAIGPTModel,
    #"gpt2": transformers.GPT2Model,
    "roberta-base": transformers.RobertaModel,
    #"roberta-large": transformers.RobertaModel,
    "distilroberta-base": transformers.RobertaModel,
    "distilbert-base-cased": transformers.DistilBertModel,
}