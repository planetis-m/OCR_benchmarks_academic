| Model | Coverage | CER | WER | OrderF1 | MathF1 | TokRecall | CharLCSRec | CostUSD | RobustAcc/$ | RobustBalanced |
|---|---|---|---|---|---|---|---|---|---|---|
| PaddlePaddle/PaddleOCR-VL-0.9B | 1.000 | 0.4634 | 0.4732 | 0.3751 | 0.7248 | 0.7678 | 0.7924 | 0.0420 | 15.826 | 0.4945 |
| allenai/olmOCR-2-7B-1025 | 1.000 | 0.4682 | 0.4893 | 0.3252 | 0.6743 | 0.8146 | 0.8496 | 0.0214 | 31.168 | 0.5232 |
| deepseek-ai/DeepSeek-OCR | 1.000 | 0.5862 | 0.6512 | 0.1452 | 0.4684 | 0.6857 | 0.7235 | 0.0041 | 124.528 | 0.6540 |

Accuracy-first (strict) winner: `PaddlePaddle/PaddleOCR-VL-0.9B`
Accuracy-first (robust) winner: `allenai/olmOCR-2-7B-1025`
Cost-first winner: `deepseek-ai/DeepSeek-OCR`
Balanced (strict) winner: `deepseek-ai/DeepSeek-OCR`
Balanced (robust) winner: `deepseek-ai/DeepSeek-OCR`
