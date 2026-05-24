from dataset import Multi30kDataset

dataset=Multi30kDataset("train")

print(len(dataset))

src,tgt=dataset[0]

print(src)
print(tgt)

print(dataset.src_vocab.stoi["<pad>"])
print(dataset.tgt_vocab.stoi["<pad>"])