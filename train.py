import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import evaluate

from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset import Multi30kDataset
from model import *
from lr_scheduler import NoamScheduler


DEVICE=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BATCH_SIZE=64

D_MODEL=512

NUM_HEADS=8

N=2

D_FF=2048

DROPOUT=0.1

EPOCHS=10

WARMUP_STEPS=4000



def collate_fn(batch):

    src_batch=[]
    tgt_batch=[]

    for src,tgt in batch:

        src_batch.append(src)

        tgt_batch.append(tgt)

    src_batch=nn.utils.rnn.pad_sequence(

        src_batch,
        batch_first=True,
        padding_value=1

    )

    tgt_batch=nn.utils.rnn.pad_sequence(

        tgt_batch,
        batch_first=True,
        padding_value=1

    )

    return src_batch,tgt_batch



train_dataset=Multi30kDataset(
    "train"
)

val_dataset=Multi30kDataset(
    "validation"
)


train_loader=DataLoader(

    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn

)

val_loader=DataLoader(

    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn

)



model=Transformer(

    len(train_dataset.src_vocab),
    len(train_dataset.tgt_vocab),

    d_model=D_MODEL,

    N=N,

    num_heads=NUM_HEADS,

    d_ff=D_FF,

    dropout=DROPOUT

).to(
    DEVICE
)


optimizer=optim.Adam(

    model.parameters(),

    betas=(0.9,0.98),

    eps=1e-9,

    lr=1

)


scheduler=NoamScheduler(

    optimizer,

    d_model=D_MODEL,

    warmup_steps=WARMUP_STEPS

)


criterion=nn.CrossEntropyLoss(

    ignore_index=1,

    label_smoothing=0.1

)


bleu=evaluate.load(
    "bleu"
)



wandb.init(

    project="DA6401_A3",

    name="TransformerBaseline"

)



best_loss=float("inf")



for epoch in range(EPOCHS):

    model.train()

    epoch_loss=0


    loop=tqdm(
        train_loader
    )


    for src,tgt in loop:

        src=src.to(
            DEVICE
        )

        tgt=tgt.to(
            DEVICE
        )

        tgt_input=tgt[:,:-1]

        tgt_output=tgt[:,1:]

        src_mask=make_src_mask(
            src,
            1
        )

        tgt_mask=make_tgt_mask(
            tgt_input,
            1
        )

        output=model(

            src,

            tgt_input,

            src_mask,

            tgt_mask

        )


        output=output.reshape(

            -1,
            output.shape[-1]

        )

        tgt_output=tgt_output.reshape(
            -1
        )


        loss=criterion(

            output,
            tgt_output
        )


        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1
        )

        optimizer.step()

        scheduler.step()

        epoch_loss+=loss.item()

        loop.set_postfix(
            loss=loss.item()
        )


    avg_loss=epoch_loss/len(
        train_loader
    )


    wandb.log(

        {

        "train_loss":avg_loss,

        "lr":optimizer.param_groups[0]["lr"]

        }

    )


    print(

        f"Epoch {epoch+1}"

    )

    print(

        f"Loss:{avg_loss:.4f}"

    )


    if avg_loss<best_loss:

        best_loss=avg_loss

        torch.save(

            model.state_dict(),

            "best_model.pth"

        )

        print(
            "Checkpoint Saved"
        )



wandb.finish()