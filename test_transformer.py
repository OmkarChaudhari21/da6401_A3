import torch

from model import *

src_vocab=10000
tgt_vocab=10000

model=Transformer(

    src_vocab,
    tgt_vocab,
    d_model=512,
    N=2,
    num_heads=8,
    d_ff=2048

)

src=torch.randint(
    0,
    1000,
    (2,15)
)

tgt=torch.randint(
    0,
    1000,
    (2,12)
)

src_mask=make_src_mask(
    src,
    pad_idx=1
)

tgt_mask=make_tgt_mask(
    tgt,
    pad_idx=1
)

out=model(
    src,
    tgt,
    src_mask,
    tgt_mask
)

print(
    out.shape
)