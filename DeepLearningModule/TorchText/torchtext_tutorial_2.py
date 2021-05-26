import spacy
from torchtext.datasets import Multi30k
from torchtext.data import Field,BucketIterator, TabularDataset

spacy_ger= spacy.load("de_core_news_sm")
space_eng = spacy.load("en_core_web_sm")

def tokenize_eng(text):
    return [tok.text for tok in space_eng.tokenizer(text)]

def tokenizr_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

english = Field(sequential= True, tokenize=tokenize_eng, lower=True, use_vocab= True)
german = Field(sequential= True, tokenize=tokenizr_ger, lower=True, use_vocab= True)

train_data, validation_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english)
)

english.build_vocab(train_data, min_freq =2, max_size = 10000)
german.build_vocab(train_data, min_freq =2, max_size = 10000)

train_iterator, validation_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data), batch_size=64, device="cpu"
)


# for batch in train_iterator:
#     print(batch.src.shape)

print(english.vocab.stoi['hello'])