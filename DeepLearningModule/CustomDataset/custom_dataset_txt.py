import os
import torch
import spacy
import pandas as pd
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: '<PAD>', 1: '<SOS>', 2: '<EOS>', 3: '<UNK>'}
        self.stoi = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sequence_list):
        frequencies = {}
        idx = 4

        for sequence in sequence_list:
            for word in self.tokenizer_eng(sequence):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

            if frequencies[word] == self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx = idx + 1

    def numericalize(self, text):
        tokenizer_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenizer_text
        ]


class FlickdrDataset(Dataset):
    def __init__(self, root_dir, caption_file, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.transform = transform
        self.dataframe = pd.read_csv(caption_file)

        self.captions = self.dataframe['caption']
        self.image = self.dataframe['image']

        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, item):
        caption_item = self.captions[item]
        image_id = self.image[item]
        img = Image.open(os.path.join(self.root_dir, image_id)).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        num_caption = [self.vocab.stoi["<SOS>"]]
        num_caption += self.vocab.numericalize(caption_item)
        num_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(num_caption)


class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [(item[0]).unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(
        root_loader,
        annotation_file,
        transform,
        batch_size=32,
        shuffle = True
):
    dataset = FlickdrDataset(root_loader, annotation_file, transform=transform)
    pad_idx = dataset.vocab.stoi['<PAD>']

    transform_new= transforms.Compose([
        transforms.ToTensor()
    ])
    loader = DataLoader(dataset, batch_size,shuffle= shuffle,
                        collate_fn=MyCollate(pad_idx=pad_idx))

    return loader


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
dataloader = get_loader(root_loader='flickr8k/images', annotation_file="flickr8k/captions.csv", transform=transform)

for idx, (img, target) in enumerate(dataloader):
    print(img.shape)
    print(target.shape)
