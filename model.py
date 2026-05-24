import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F



#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot Product Attention
    """

    d_k = Q.shape[-1]

    scores = torch.matmul(
        Q,
        K.transpose(-2,-1)
    ) / math.sqrt(d_k)

    if mask is not None:

        scores = scores.masked_fill(
            mask,
            -1e9
        )

    attention_weights = torch.softmax(
        scores,
        dim=-1
    )

    output = torch.matmul(
        attention_weights,
        V
    )

    return output, attention_weights


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build source padding mask
    """

    mask = (src == pad_idx)

    mask = mask.unsqueeze(1).unsqueeze(2)

    return mask

def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    batch_size=tgt.shape[0]

    tgt_len=tgt.shape[1]

    pad_mask=(

        tgt==pad_idx

    ).unsqueeze(1).unsqueeze(2)

    future_mask=torch.triu(

        torch.ones(
            tgt_len,
            tgt_len,
            device=tgt.device
        ),

        diagonal=1

    ).bool()

    future_mask=future_mask.unsqueeze(
        0
    ).unsqueeze(
        1
    )

    mask=pad_mask | future_mask

    return mask

#  MULTI-HEAD ATTENTION 


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model:int,
        num_heads:int,
        dropout:float=0.1
    )->None:

        super().__init__()

        assert d_model % num_heads ==0

        self.d_model=d_model
        self.num_heads=num_heads
        self.d_k=d_model//num_heads

        self.W_q=nn.Linear(
            d_model,
            d_model
        )

        self.W_k=nn.Linear(
            d_model,
            d_model
        )

        self.W_v=nn.Linear(
            d_model,
            d_model
        )

        self.W_o=nn.Linear(
            d_model,
            d_model
        )

        self.dropout=nn.Dropout(
            dropout
        )


    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        batch_size=query.shape[0]

        Q=self.W_q(query)
        K=self.W_k(key)
        V=self.W_v(value)

        Q=Q.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1,2)

        K=K.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1,2)

        V=V.view(
            batch_size,
            -1,
            self.num_heads,
            self.d_k
        ).transpose(1,2)

        x,attn=scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        x=x.transpose(
            1,
            2
        ).contiguous()

        x=x.view(
            batch_size,
            -1,
            self.d_model
        )

        x=self.W_o(x)

        return x



#   POSITIONAL ENCODING  


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):

        super().__init__()

        self.dropout=nn.Dropout(
            dropout
        )

        pe=torch.zeros(
            max_len,
            d_model
        )

        position=torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term=torch.exp(

            torch.arange(
                0,
                d_model,
                2
            )

            *(-math.log(10000.0)/d_model)

        )

        pe[:,0::2]=torch.sin(
            position*div_term
        )

        pe[:,1::2]=torch.cos(
            position*div_term
        )

        pe=pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )


    def forward(
        self,
        x
    ):

        x=x+self.pe[:,:x.shape[1]]

        return self.dropout(
            x
        )



#  FEED-FORWARD NETWORK 


class PositionwiseFeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.linear1=nn.Linear(
            d_model,
            d_ff
        )

        self.linear2=nn.Linear(
            d_ff,
            d_model
        )

        self.dropout=nn.Dropout(
            dropout
        )


    def forward(
        self,
        x
    ):

        x=self.linear1(x)

        x=torch.relu(x)

        x=self.dropout(x)

        x=self.linear2(x)

        return x



#  ENCODER LAYER  


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.attn=MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn=PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(
            d_model
        )

        self.norm2=nn.LayerNorm(
            d_model
        )

        self.dropout=nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        src_mask
    ):

        attn=self.attn(
            x,
            x,
            x,
            src_mask
        )

        x=self.norm1(
            x+self.dropout(attn)
        )

        ff=self.ffn(x)

        x=self.norm2(
            x+self.dropout(ff)
        )

        return x


#   DECODER LAYER 


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.self_attn=MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.cross_attn=MultiHeadAttention(
            d_model,
            num_heads,
            dropout
        )

        self.ffn=PositionwiseFeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(
            d_model
        )

        self.norm2=nn.LayerNorm(
            d_model
        )

        self.norm3=nn.LayerNorm(
            d_model
        )

        self.dropout=nn.Dropout(
            dropout
        )


    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        attn=self.self_attn(
            x,
            x,
            x,
            tgt_mask
        )

        x=self.norm1(
            x+self.dropout(attn)
        )

        attn=self.cross_attn(
            x,
            memory,
            memory,
            src_mask
        )

        x=self.norm2(
            x+self.dropout(attn)
        )

        ff=self.ffn(x)

        x=self.norm3(
            x+self.dropout(ff)
        )

        return x



#  ENCODER & DECODER STACKS


class Encoder(nn.Module):

    def __init__(
        self,
        layer,
        N
    ):

        super().__init__()

        self.layers=nn.ModuleList(
            [
                copy.deepcopy(layer)
                for _ in range(N)
            ]
        )

        self.norm=nn.LayerNorm(
            layer.norm1.normalized_shape
        )


    def forward(
        self,
        x,
        mask
    ):

        for layer in self.layers:

            x=layer(
                x,
                mask
            )

        return self.norm(x)


class Decoder(nn.Module):

    def __init__(
        self,
        layer,
        N
    ):

        super().__init__()

        self.layers=nn.ModuleList(
            [
                copy.deepcopy(layer)
                for _ in range(N)
            ]
        )

        self.norm=nn.LayerNorm(
            layer.norm1.normalized_shape
        )


    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        for layer in self.layers:

            x=layer(
                x,
                memory,
                src_mask,
                tgt_mask
            )

        return self.norm(x)



#   FULL TRANSFORMER  


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size:int,
        tgt_vocab_size:int,
        d_model:int=512,
        N:int=6,
        num_heads:int=8,
        d_ff:int=2048,
        dropout:float=0.1,
        checkpoint_path:str=None,
    )->None:

        super().__init__()

        self.d_model=d_model

        self.src_embedding=nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding=nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        self.positional_encoding=PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer=EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        decoder_layer=DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder=Encoder(
            encoder_layer,
            N
        )

        self.decoder=Decoder(
            decoder_layer,
            N
        )

        self.fc_out=nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self.dropout=nn.Dropout(
            dropout
        )

        for p in self.parameters():

            if p.dim()>1:

                nn.init.xavier_uniform_(p)

        if checkpoint_path is not None:

            gdown.download(
                id="<.pth drive id>",
                output=checkpoint_path,
                quiet=False
            )

            self.load_state_dict(

                torch.load(
                    checkpoint_path,
                    map_location="cpu"
                )
            )


    def encode(
        self,
        src:torch.Tensor,
        src_mask:torch.Tensor,
    )->torch.Tensor:


        src_embed=self.src_embedding(
            src
        )*math.sqrt(
            self.d_model
        )

        src_embed=self.positional_encoding(
            src_embed
        )

        memory=self.encoder(
            src_embed,
            src_mask
        )

        return memory


    def decode(
        self,
        memory:torch.Tensor,
        src_mask:torch.Tensor,
        tgt:torch.Tensor,
        tgt_mask:torch.Tensor,
    )->torch.Tensor:


        tgt_embed=self.tgt_embedding(
            tgt
        )*math.sqrt(
            self.d_model
        )

        tgt_embed=self.positional_encoding(
            tgt_embed
        )

        decoder_output=self.decoder(
            tgt_embed,
            memory,
            src_mask,
            tgt_mask
        )

        logits=self.fc_out(
            decoder_output
        )

        return logits


    def forward(
        self,
        src:torch.Tensor,
        tgt:torch.Tensor,
        src_mask:torch.Tensor,
        tgt_mask:torch.Tensor,
    )->torch.Tensor:


        memory=self.encode(
            src,
            src_mask
        )

        logits=self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

        return logits


    def infer(
        self,
        src_sentence:str
    )->str:

        self.eval()

        device=next(
            self.parameters()
        ).device

        try:

            tokens=src_sentence.lower().split()

        except:

            return ""

        vocab={

            "<pad>":1,
            "<sos>":2,
            "<eos>":3,
            "<unk>":0

        }

        src_indices=[

            vocab["<sos>"]

        ]

        src_indices.extend(

            [

            vocab.get(
                token,
                vocab["<unk>"]
            )

            for token in tokens

            ]

        )

        src_indices.append(

            vocab["<eos>"]

        )

        src=torch.tensor(

            src_indices

        ).unsqueeze(
            0
        ).to(
            device
        )

        src_mask=make_src_mask(
            src,
            pad_idx=1
        )

        memory=self.encode(
            src,
            src_mask
        )

        ys=torch.tensor(

            [[2]]

        ).to(
            device
        )

        max_len=50

        for i in range(max_len):

            tgt_mask=make_tgt_mask(
                ys,
                pad_idx=1
            )

            out=self.decode(

                memory,

                src_mask,

                ys,

                tgt_mask

            )

            prob=out[:,-1]

            next_word=torch.argmax(
                prob,
                dim=1
            )

            ys=torch.cat(

                [

                ys,

                next_word.unsqueeze(0)

                ],

                dim=1

            )

            if next_word.item()==3:
                break


        output=[]

        for idx in ys.squeeze():

            idx=idx.item()

            if idx in [1,2,3]:
                continue

            output.append(
                str(idx)
            )

        return " ".join(
            output
        )