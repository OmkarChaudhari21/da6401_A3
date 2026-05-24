import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import spacy
from collections import Counter
from tqdm import tqdm


class Vocabulary:

    def __init__(self,freq_threshold=2):

        self.freq_threshold = freq_threshold

        self.itos = {
            0:"<unk>",
            1:"<pad>",
            2:"<sos>",
            3:"<eos>"
        }

        self.stoi = {
            "<unk>":0,
            "<pad>":1,
            "<sos>":2,
            "<eos>":3
        }

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self,sentences,tokenizer):

        frequencies=Counter()

        idx=4

        for sentence in tqdm(sentences):

            for word in tokenizer(sentence):

                frequencies[word]+=1

                if frequencies[word]==self.freq_threshold:

                    self.stoi[word]=idx
                    self.itos[idx]=word

                    idx+=1

    def numericalize(self,text,tokenizer):

        tokenized=tokenizer(text)

        return [

            self.stoi[token]

            if token in self.stoi
            else self.stoi["<unk>"]

            for token in tokenized
        ]


class Multi30kDataset(Dataset):

    def __init__(self,split='train'):

        self.split=split

        self.data=load_dataset(
            "bentrevett/multi30k"
        )[split]

        self.spacy_de=spacy.load(
            "de_core_news_sm"
        )

        self.spacy_en=spacy.load(
            "en_core_web_sm"
        )

        self.src_vocab=Vocabulary()
        self.tgt_vocab=Vocabulary()

        self.build_vocab()

        self.processed_data=self.process_data()


    def tokenize_de(self,text):

        return [

            token.text.lower()

            for token in self.spacy_de.tokenizer(text)
        ]


    def tokenize_en(self,text):

        return [

            token.text.lower()

            for token in self.spacy_en.tokenizer(text)
        ]


    def build_vocab(self):

        src_sentences=self.data["de"]

        tgt_sentences=self.data["en"]

        print("Building German vocabulary")

        self.src_vocab.build_vocabulary(
            src_sentences,
            self.tokenize_de
        )

        print("Building English vocabulary")

        self.tgt_vocab.build_vocabulary(
            tgt_sentences,
            self.tokenize_en
        )


    def process_data(self):

        processed=[]

        for item in tqdm(self.data):

            src_tokens=[

                self.src_vocab.stoi["<sos>"]

            ]

            src_tokens+=self.src_vocab.numericalize(
                item["de"],
                self.tokenize_de
            )

            src_tokens.append(
                self.src_vocab.stoi["<eos>"]
            )


            tgt_tokens=[

                self.tgt_vocab.stoi["<sos>"]

            ]

            tgt_tokens+=self.tgt_vocab.numericalize(
                item["en"],
                self.tokenize_en
            )

            tgt_tokens.append(
                self.tgt_vocab.stoi["<eos>"]
            )

            processed.append(
                (
                    torch.tensor(src_tokens),
                    torch.tensor(tgt_tokens)
                )
            )

        return processed


    def __len__(self):

        return len(self.processed_data)


    def __getitem__(self,index):

        return self.processed_data[index]