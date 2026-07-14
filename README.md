# Indic Spoken Language Identification (SLID)

Fine-tuning a multilingual speech foundation model to identify which Indian language is being spoken in an audio clip, with a custom audio augmentation pipeline to improve robustness.

Built as a deep learning project at Saarland University (Neural Networks: Theory and Implementation).

---
**Authors:** Akash Chavan, Vaishnavi, Gauhar Khan

## Overview

Spoken Language Identification (SLID) is the task of predicting the language of an utterance directly from raw audio. This project fine-tunes a pretrained multilingual speech model for SLID across Indic languages and studies whether **waveform-level augmentation** improves classification performance.

**Key idea:** pretrained speech foundation models already encode rich acoustic representations, but they are sensitive to recording conditions. Augmenting the training audio (pitch, noise, masking) exposes the model to more acoustic variation and improves generalisation.

---

## Approach

**Base model:** [`facebook/mms-300m`](https://huggingface.co/facebook/mms-300m) — Meta's Massively Multilingual Speech encoder (300M params), used as an audio classification backbone.

Also evaluated:
- `utter-project/mHuBERT-147`
- `facebook/wav2vec2-xls-r-300m`

**Dataset:** `badrex/nnti-dataset-full` (Hugging Face), 16 kHz audio, multi-class Indic language labels.

**Augmentation pipeline** — each transform applied stochastically (p = 0.5) during training only:

| Transform | Detail |
|---|---|
| Pitch shift | ±3 semitones (`torchaudio.transforms.PitchShift`) |
| Gaussian noise | Additive, scale sampled from `U(0.001, 0.01)` |
| Time masking | Contiguous span of 5–15% of the waveform zeroed out |

Augmentation is applied to the raw waveform *before* feature extraction, so the model sees genuinely different acoustic inputs each epoch.

---

## Training setup

| Setting | Value |
|---|---|
| Objective | Multi-class audio classification (`AutoModelForAudioClassification`) |
| Optimiser | AdamW, LR `2e-5` |
| Schedule | Cosine decay, 15% warmup |
| Batch size | 8 (× 4 gradient accumulation → effective 32) |
| Epochs | 20 |
| Weight decay | 0.05 |
| Model selection | Best checkpoint on validation loss |
| Tracking | Weights & Biases (`Indic-SLID` project) |

Trained on the Saarland University CS GPU cluster (1× GPU, 4 CPUs, 16 GB RAM) via **HTCondor**, using a **Docker** image for a reproducible environment.

---

## Evaluation

- **Confusion matrix** — per-language precision/recall, showing which language pairs the model confuses.
- **t-SNE projection** of the learned embeddings — visualising whether languages form separable clusters in representation space.

Both artefacts are written to `indic-SLID/inprogress/` after a run.

---

## Repository structure

```
.
├── audio_ml.py        # Full pipeline: data loading, augmentation, fine-tuning, evaluation
├── audio_ml.sub       # HTCondor submit file (GPU job, Docker universe)
├── Dockerfile.txt     # Container definition for the training environment
└── README.md
```

---

## Running it

### Locally

```bash
pip install torch torchaudio transformers datasets wandb scikit-learn matplotlib seaborn
python audio_ml.py
```

Set `HF_TOKEN` and `WANDB_API_KEY` as environment variables if you want dataset access and experiment tracking.

### On an HTCondor cluster

```bash
mkdir -p runlogs
condor_submit audio_ml.sub

condor_q                       # job status
tail -f runlogs/audio_ml.*.log # live logs
```

Outputs (trained model, `confusion_matrix.png`, `tsne.png`) are transferred back to `indic-SLID/inprogress/` on completion.

---

## Tech stack

`Python` · `PyTorch` · `torchaudio` · `Hugging Face Transformers` · `Datasets` · `Weights & Biases` · `scikit-learn` · `HTCondor` · `Docker`
