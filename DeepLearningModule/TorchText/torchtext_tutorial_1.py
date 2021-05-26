# Step:
# 1.Specify pre processing should be done -> Field
# 2.Use Dataset to load data -> TabularDataset
# 3.Construct an iterator to do batching & padding -> BucketIterator

from torchtext.data import Field, TabularDataset, BucketIterator
import spacy

spacy_en = spacy.load("en_core_web_sm")


def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'quote': ('q', quote), 'score': ('s', score)}

train_data, test_data = TabularDataset.splits(
    path='mydata/mydata',
    train='train.json',
    test='test.json',
    format='json',
    fields=fields
)

quote.build_vocab(train_data, max_size =10000, min_freq= 1)

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size= 2,
    device = 'cpu',
    # vectors = 'glove.6B.100'
)

for batch in train_iterator:
    print(batch.s)
    print(batch.q)