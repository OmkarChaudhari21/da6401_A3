import torch

from model import *

x=torch.rand(
    2,
    10,
    512
)

mask=torch.zeros(
    2,
    1,
    1,
    10
).bool()

layer=EncoderLayer(
    512,
    8,
    2048
)

out=layer(
    x,
    mask
)

print(out.shape)