import transformers


MAX_LEN = 256
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 8
EPOCHS = 10
BERT_PATH = "../input/bert-base-uncased/"
MODEL_PATH = "model.bin"
#TRAINING_FILE = "../input/imdb.csv"
TRAINING_FILE = "../tweet_input/tweet_input.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    BERT_PATH,
    do_lower_case=True
)
