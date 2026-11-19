# Task 1: LoRA Fine-Tuning for Emotion Classification

## 1. Project Overview

This project fine-tunes the Qwen2.5-0.5B-Instruct model for multi-class emotion classification using Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

The objective is to compare the baseline model against the fine-tuned model using a held-out test dataset and evaluate its classification performance.

## 2. Model and Dataset

- **Base Model:** Qwen/Qwen2.5-0.5B-Instruct
- **Dataset:** dair-ai/emotion
- **Task:** Emotion Classification
- **Number of Classes:** 6
- **Training Samples:** 16,000
- **Validation Samples:** 2,000
- **Test Samples:** 2,000

Emotion labels:
- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

## 3. Fine-Tuning Method

LoRA (Low-Rank Adaptation) was used to reduce the number of trainable parameters while keeping the pretrained model weights frozen.

### LoRA Configuration

| Parameter | Value |
|---|---|
| Rank (r) | 8 |
| Alpha | 16 |
| Dropout | 0.1 |
| Target Modules | q_proj, v_proj |
| Modules to Save | score |
| Epochs | 2 |
| Learning Rate | 0.0002 |
| Batch Size | 8 |
| Gradient Accumulation | 2 |
| Maximum Sequence Length | 128 |
| Precision | FP16 |
| Random Seed | 42 |

Trainable parameters: 546,048 out of 494,584,192 total parameters (approximately 0.1104%).

## 4. Evaluation Results

| Metric | Baseline | Fine-Tuned |
|---|---:|---:|
| Accuracy | 0.3310 | 0.9265 |
| Macro F1 | 0.1140 | 0.8841 |

The fine-tuned model achieved 92.65% test accuracy and 0.8841 Macro F1.

The baseline represents the pretrained backbone with an untrained classification head, not a prompted zero-shot classifier.

## 5. Error Analysis

Three representative misclassification examples:

1. "i feel underappreciated and under valued" — Actual: Joy; Predicted: Sadness.
2. A statement describing workplace conflict, unfair treatment, and feeling terrible — Actual: Anger; Predicted: Sadness.
3. "i have reported feeling marginalized intimidated and or subjected to threats of retaliation" — Actual: Fear; Predicted: Sadness.

These errors indicate that some negative emotional expressions overlap and may be classified as sadness.

## 6. Estimated GPU Memory for a 7B Model

Illustrative memory estimates:

| Fine-Tuning Method | Estimated Memory |
|---|---:|
| Full Fine-Tuning | 112 GB |
| LoRA | 15.12 GB |
| QLoRA | 4.62 GB |

These are simplified parameter and optimizer-state estimates. They exclude activations, temporary buffers, memory fragmentation, and framework overhead. Actual requirements depend on training configuration.

## 7. Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- PEFT
- LoRA
- Google Colab

## 8. Project Structure

```text
task1-finetuning/
├── notebooks/
│   ├── environment_check.ipynb
│   └── task1_lora_finetuning.ipynb
├── results/
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── metrics.json
│   └── tokenizer files
└── README.md
```

## 9. Conclusion

LoRA fine-tuning significantly improved emotion classification performance compared with the baseline. The experiment demonstrates how parameter-efficient fine-tuning can adapt a pretrained language model while updating only a small proportion of its parameters.