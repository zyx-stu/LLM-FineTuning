# LLM-FineTuning
In this project, I fine-tuned a Large Language Model (LLM) for both text classification and instruction-following tasks.  The workflow includes data preprocessing, model training, fine-tuning on custom datasets, and evaluation through result visualization and performance analysis.
# 🔧 LLM Fine-tuning from Scratch

A hands-on implementation of two core fine-tuning approaches for GPT-2 using PyTorch — **Classification Fine-tuning** (spam detection) and **Instruction Fine-tuning** (Alpaca-style SFT). Both notebooks are fully self-contained and ready to run.

---

## 📌 Project Overview

This project demonstrates the two most common fine-tuning paradigms for Large Language Models:

| Approach | Notebook | Task | Model | Dataset |
|---|---|---|---|---|
| **Classification** | `01_classification_finetuning.ipynb` | Spam detection | GPT-2 Small (124M) | SMS Spam Collection |
| **Instruction SFT** | `02_instruction_finetuning.ipynb` | Follow instructions | GPT-2 Medium (355M) | Alpaca-style (1,100 pairs) |

---

## 🗂️ Repository Structure

```
LLM-finetuning/
│
├── 01_classification_finetuning.ipynb   # Binary spam classifier via classification head
├── 02_instruction_finetuning.ipynb      # Instruction following via SFT (Alpaca format)
│
├── gpt_download3.py                     # GPT-2 weight downloader (auto-fetched in notebooks)
│
├── # ── Generated at runtime ──
├── sms_spam_collection/                 # SMS dataset (auto-downloaded)
├── gpt2/                                # GPT-2 weights (auto-downloaded)
├── instruction-data.json                # Alpaca instruction pairs (auto-downloaded)
├── train.csv / validation.csv / test.csv
├── spam_classifier.pth                  # Saved classification model
├── gpt2medium355M-sft.pth               # Saved instruction fine-tuned model
├── test_responses.json                  # Generated model responses on test set
└── README.md
```

---

## 📒 Notebook Breakdown

---

### `01_classification_finetuning.ipynb` — Spam Classifier

Fine-tune GPT-2 Small (124M) to classify SMS messages as spam or not spam.

**Section 1 — Setup:** Full GPT-2 architecture + pretrained weights loaded via OpenAI checkpoint.

**Section 2 — Dataset:** Download SMS Spam Collection, balance classes (747 each), 70/10/20 split.

**Section 3 — SpamDataset & DataLoaders:** Custom `SpamDataset` — tokenize + pad. DataLoader with batch size 8.

**Section 4 — Classification Head:** Freeze all layers → replace 50,257-dim head with 2-dim classifier → unfreeze last transformer block + LayerNorm.

**Section 5 — Training & Evaluation:** 5 epochs with AdamW (lr=5e-5). Tracks cross-entropy loss and classification accuracy.

**Section 6 — Inference & Save:** `classify_review()` for end-to-end inference. Save/load via `torch.save`.

#### Results

| Split | Accuracy |
|-------|----------|
| Train | ~98% |
| Val   | ~97% |
| Test  | ~96% |

---

### `02_instruction_finetuning.ipynb` — Instruction SFT

Supervised Fine-Tuning (SFT) of GPT-2 Medium (355M) on 1,100 Alpaca-style instruction-response pairs.

**Section 1 — Setup:** Full GPT-2 architecture + GPT-2 Medium (355M) pretrained weights.

**Section 2 — Instruction Dataset:** Auto-downloads Alpaca-format JSON. Formats entries with `### Instruction / ### Input / ### Response` template. 85/10/5 split.

**Section 3 — Custom Collate & DataLoaders:** `InstructionDataset` + `custom_collate_fn` with dynamic per-batch padding. Padding tokens masked with `-100` so cross-entropy ignores them.

**Section 4 — Baseline Evaluation:** Shows pretrained model failing to follow instructions before SFT.

**Section 5 — Training:** Full fine-tune, 2 epochs, AdamW (lr=0.0005). Prints sample output after each epoch.

**Section 6 — Extract & Save Responses:** Generates and saves test responses to `test_responses.json`. Saves model weights.

**Section 7 — Automated Evaluation:** Uses **Llama-3 8B Instruct** (Hugging Face) as a judge to score generated responses on 0–100.

---

## 🚀 Getting Started

```bash
pip install torch tiktoken tensorflow tqdm numpy pandas matplotlib
pip install transformers accelerate huggingface_hub  # for Section 7 only

git clone https://github.com/your-username/LLM-finetuning.git
cd LLM-finetuning
jupyter notebook
```

---

## 🏗️ Architecture Details

### Classification Fine-tuning

| Component | Specification |
|---|---|
| Base model | GPT-2 Small |
| Parameters | 124M (pretrained) |
| Trainable params | ~7M (last block + head) |
| Classifier head | Linear(768 → 2) |
| Loss | Cross-entropy (last token) |
| Optimizer | AdamW, lr=5e-5 |
| Epochs | 5 |

### Instruction Fine-tuning (SFT)

| Component | Specification |
|---|---|
| Base model | GPT-2 Medium |
| Parameters | 355M |
| Trainable params | All 355M (full fine-tune) |
| Loss | Cross-entropy, ignore_index=-100 |
| Optimizer | AdamW, lr=0.0005 |
| Epochs | 2 (default) |
| Prompt format | Alpaca |

---

## 🔑 Key Concepts Covered

**Classification Fine-tuning:** Transfer learning from LM to classifier, selective layer freezing, last-token classification signal, balanced datasets.

**Instruction Fine-tuning:** Alpaca prompt formatting, dynamic per-batch padding, target masking, SFT training loop, LLM-as-a-judge evaluation.

---

## 💡 Tips for Better Results

| Goal | What to change |
|---|---|
| Higher classification accuracy | Unfreeze more transformer blocks |
| Better instruction following | Increase SFT epochs (3–5) or use more data |
| Faster training | Use GPU (cuda) or Apple Silicon (mps) |
| Larger scale | Switch to gpt2-medium or gpt2-xl |
| Production use | Replace GPT-2 with Llama-3, Mistral, or Phi-3 |

---

## 📄 Acknowledgements

- GPT-2 weights by [OpenAI](https://github.com/openai/gpt-2)
- Instruction dataset from [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- SMS Spam Collection from [UCI ML Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)
- Inspired by *"Build a Large Language Model (From Scratch)"* by Sebastian Raschka

---

## 📝 License

Educational use. GPT-2 weights subject to [OpenAI's usage policy](https://github.com/openai/gpt-2/blob/master/model_card.md). Llama-3 requires [Meta's license](https://huggingface.co/meta-llama/Meta-Llama-3-8B).
