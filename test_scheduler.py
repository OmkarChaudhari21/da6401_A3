from lr_scheduler import get_lr_history

history=get_lr_history(
    d_model=512,
    warmup_steps=4000,
    total_steps=10000
)

print(history[:10])

print(max(history))