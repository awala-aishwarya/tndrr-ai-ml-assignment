# Task 3: Local LLM Optimization

## 1. Project Overview

This project evaluates Large Language Model inference under different precision settings.

The objective is to compare FP16 and 4-bit quantization in terms of GPU memory usage, inference speed, and generated response quality.

The experiments use the Qwen2.5-0.5B-Instruct model on an NVIDIA Tesla T4 GPU through Kaggle.

## 2. Model and Environment

- **Model:** Qwen/Qwen2.5-0.5B-Instruct
- **GPU:** NVIDIA Tesla T4
- **Platform:** Kaggle Notebook
- **Framework:** PyTorch
- **Libraries:** Transformers, Accelerate, BitsAndBytes
- **Prompt Set:** 10 identical prompts for both configurations
- **Generation:** Deterministic generation (`do_sample=False`)
- **Maximum New Tokens:** 100

## 3. Experimental Configurations

Two configurations were evaluated:

### Configuration 1: FP16

The model was loaded using half-precision floating-point weights.

FP16 uses 16 bits per weight value and generally offers a balance between numerical precision and GPU memory consumption.

### Configuration 2: 4-bit NF4

The model was loaded using 4-bit NormalFloat quantization.

Configuration:
- 4-bit loading enabled.
- Quantization type: NF4.
- Compute dtype: FP16.
- Double quantization enabled.

This configuration reduces model-weight memory requirements.

## 4. Performance Results

| Metric | FP16 | 4-bit NF4 |
|---|---:|---:|
| Peak GPU Memory | 0.934 GB | 0.445 GB |
| Throughput | 35.59 tokens/s | 20.58 tokens/s |
| Generated Tokens in Timed Run | 100 | 86 |
| Elapsed Time | 2.81 seconds | 4.179 seconds |

### Observations

- 4-bit NF4 reduced peak allocated GPU memory by approximately 52.4%.
- FP16 achieved higher measured throughput in this experiment.
- 4-bit quantization reduced memory consumption but introduced additional inference overhead.
- The two timed runs generated different token counts, so throughput values should be interpreted as measurements from this specific benchmark.

The saved quality outputs compare both configurations using the same ten prompts.

## 5. KV Cache

KV cache stores attention key and value tensors from previously processed tokens.

During autoregressive generation, these cached values avoid recomputing attention information for the entire previous sequence.

**Benefits:**
- Reduces repeated computation.
- Improves token-generation efficiency.
- Supports faster autoregressive decoding.

**Limitation:**

KV cache consumes GPU memory. Its memory usage increases with the number of active sequences, model layers, and context length.

## 6. Quantization

Quantization reduces the numerical precision used to represent model weights.

- FP16 uses 16-bit floating-point values.
- 4-bit NF4 represents weights using a lower-bit format.

**Advantages of quantization:**
- Reduces GPU memory requirements.
- Makes larger models easier to load on limited hardware.
- Can reduce storage requirements.

**Trade-offs:**
- May affect model output quality.
- Can introduce additional computation overhead.
- Does not always improve inference speed.

## 7. 20 Concurrent Users on a 24 GB GPU

An illustrative estimate for serving a 7B model using 4-bit quantization:

| Component | Estimated Memory |
|---|---:|
| Quantized Model Weights | 3.5 GB |
| Quantization Metadata | 0.5 GB |
| Runtime and Workspace | 3 GB |
| KV Cache | 5.63 GB |
| **Total** | **12.63 GB** |

Estimated remaining memory: approximately 11.37 GB.

KV-cache assumptions:
- 32 transformer layers.
- 8 KV heads.
- Head dimension 128.
- FP16 KV cache.
- 2,304 tokens per session.
- 20 concurrent sessions.

This is a theoretical capacity estimate, not a guarantee of serving throughput or latency. Actual usage depends on batching, sequence lengths, framework overhead, and scheduling.

## 8. Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Accelerate
- BitsAndBytes
- Kaggle
- Qwen2.5-0.5B-Instruct
- FP16
- NF4 Quantization

## 9. Project Structure

```text
task3-local-llm/
├── results/
│   ├── task3_quality_fp16.json
│   └── task3_quality_4bit.json
├── task3-model-optimization.ipynb
├── prompts/
├── benchmarks/
└── README.md
```

## 10. Conclusion

The experiment demonstrates the trade-off between inference speed and memory usage.

FP16 achieved higher measured throughput, while 4-bit NF4 reduced GPU memory consumption by approximately 52.4%. The results show how quantization can help deploy language models in environments with limited GPU resources.