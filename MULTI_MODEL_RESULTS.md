# PRISM Multi-Model Benchmark

**Date:** 2026-06-26 00:17
**Hardware:** Apple M1, 8 GB RAM
**Platform:** macOS-15.7.3-arm64-arm-64bit

---

## Summary

| Model | Size | Load | avg TPS | avg RAM |
|-------|------|------|---------|---------|
| Gemma 3 4B INT4 | 3.3 GB | 47153ms | **18.9** | 36.0 MB |
| Mistral 7B INT4 | 4.1 GB | 1655193ms | **9.9** | 31.0 MB |
| Llama 3.1 8B INT4 | 4.7 GB | 1821610ms | **7.3** | 32.0 MB |

---

## Per-Prompt Detail

### Gemma 3 4B INT4

| Prompt | Tier | TPS | Tokens | Time | RAM |
|--------|------|-----|--------|------|-----|
| simple | simple | 14.8 | 129 | 8.727s | 37.0 MB |
| medium | medium | 21.5 | 513 | 23.812s | 37.0 MB |
| complex | complex | 20.3 | 1025 | 50.429s | 35.0 MB |

### Mistral 7B INT4

| Prompt | Tier | TPS | Tokens | Time | RAM |
|--------|------|-----|--------|------|-----|
| simple | simple | 4.0 | 15 | 3.791s | 33.0 MB |
| medium | medium | 13.2 | 512 | 38.922s | 31.0 MB |
| complex | complex | 12.4 | 609 | 49.078s | 28.0 MB |

### Llama 3.1 8B INT4

| Prompt | Tier | TPS | Tokens | Time | RAM |
|--------|------|-----|--------|------|-----|
| simple | simple | 2.5 | 128 | 52.209s | 38.0 MB |
| medium | medium | 11.4 | 511 | 44.819s | 23.0 MB |
| complex | complex | 8.1 | 1023 | 126.559s | 36.0 MB |
