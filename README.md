# DA6401 Assignment 3

## Transformer from Scratch for German → English Machine Translation

**Name:** Omkar Ashok Chaudhari  
**Roll Number:** NA22B059  

---

## Overview

This project implements a complete Transformer architecture from scratch using PyTorch for German → English machine translation on the Multi30k dataset.

The implementation follows the paper:

Attention Is All You Need (Vaswani et al., 2017)

No built-in Transformer modules such as:

- torch.nn.Transformer
- torch.nn.MultiheadAttention

were used.

---

## Features Implemented

- Scaled Dot Product Attention
- Multi-Head Attention
- Positional Encoding
- Encoder Layer
- Decoder Layer
- Encoder Stack
- Decoder Stack
- Source and Target Masking
- Feed Forward Network
- Noam Learning Rate Scheduler
- Label Smoothing
- Gradient Clipping
- Greedy Decoding
- Weights & Biases Integration
- Model Checkpoint Saving

---

## Dataset

Dataset used:

Multi30k German-English Translation Dataset

Dataset statistics:

| Split | Samples |
|---------|---------|
| Train | 29000 |
| Validation | 1014 |
| Test | 1000 |

Tokenization:

- German: spaCy `de_core_news_sm`
- English: spaCy `en_core_web_sm`

Special tokens:

- `<unk>`
- `<pad>`
- `<sos>`
- `<eos>`

---

## Project Structure

```text
da6401_A3/
│
├── dataset.py
├── model.py
├── lr_scheduler.py
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── wandb/
├── venv/
│
└── best_model.pth
```

---

## Installation

Create virtual environment:

```bash
python -m venv venv
```

Activate:

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download spaCy models:

```bash
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

---

## Training

Run:

```bash
python train.py
```

---

## Hyperparameters

| Parameter | Value |
|------------|--------|
| Batch Size | 64 |
| Epochs | 10 |
| d_model | 512 |
| Heads | 8 |
| Layers | 2 |
| Feed Forward Dimension | 2048 |
| Dropout | 0.1 |
| Warmup Steps | 4000 |

---

## Training Results

| Epoch | Training Loss |
|---------|---------------|
| 1 | 5.9965 |
| 2 | 4.0943 |
| 3 | 3.3938 |
| 4 | 2.9763 |
| 5 | 2.7111 |
| 6 | 2.5438 |
| 7 | 2.4282 |
| 8 | 2.3451 |
| 9 | 2.2874 |
| 10 | 2.2045 |

Final Training Loss:

```text
2.2045
```

---

## Observations

- Training loss decreased consistently.
- No exploding gradients observed.
- Noam scheduler produced stable training.
- W&B logging successfully tracked metrics.
- Model checkpoint saving worked correctly.

---

## W&B Dashboard

Training metrics tracked:

- Training Loss
- Learning Rate
- Checkpoints

W&B Project:

Add project link here.

---

## Future Improvements

- BLEU evaluation
- Beam Search decoding
- Attention visualization
- Hyperparameter tuning
- Larger Transformer architecture
- Learned positional embeddings

---

## References

1. Vaswani et al., Attention Is All You Need  
   https://arxiv.org/abs/1706.03762

2. Multi30k Dataset  
   https://huggingface.co/datasets/bentrevett/multi30k

3. PyTorch Documentation  
   https://pytorch.org/docs/

---