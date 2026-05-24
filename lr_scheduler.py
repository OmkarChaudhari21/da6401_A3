"""
Noam Learning Rate Scheduler
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler


class NoamScheduler(LRScheduler):

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int=-1
    ):

        self.d_model=d_model
        self.warmup_steps=warmup_steps

        super().__init__(
            optimizer,
            last_epoch
        )


    def _get_lr_scale(self):

        step=max(
            self.last_epoch+1,
            1
        )

        scale=(self.d_model**(-0.5))*min(
            step**(-0.5),
            step*(self.warmup_steps**(-1.5))
        )

        return scale


    def get_lr(self):

        scale=self._get_lr_scale()

        return [

            base_lr*scale

            for base_lr in self.base_lrs
        ]


def get_lr_history(
    d_model,
    warmup_steps,
    total_steps
):

    dummy_model=torch.nn.Linear(
        1,
        1
    )

    optimizer=optim.Adam(
        dummy_model.parameters(),
        lr=1.0
    )

    scheduler=NoamScheduler(
        optimizer,
        d_model,
        warmup_steps
    )

    history=[]

    for _ in range(total_steps):

        history.append(
            optimizer.param_groups[0]["lr"]
        )

        optimizer.step()

        scheduler.step()

    return history


if __name__=="__main__":

    import matplotlib.pyplot as plt

    D_MODEL=512
    WARMUP_STEPS=4000
    TOTAL_STEPS=20000

    lrs=get_lr_history(
        D_MODEL,
        WARMUP_STEPS,
        TOTAL_STEPS
    )

    plt.figure(figsize=(10,5))

    plt.plot(lrs)

    plt.axvline(
        WARMUP_STEPS,
        color='red',
        linestyle='--'
    )

    plt.xlabel("Step")
    plt.ylabel("Learning Rate")

    plt.title(
        "Noam LR Scheduler"
    )

    plt.show()