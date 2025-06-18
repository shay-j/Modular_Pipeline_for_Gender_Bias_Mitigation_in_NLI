# Modular Pipeline for Gender Bias Mitigation in NLI

## 📌 Problem Statement
Natural Language Inference (NLI) models often infer gender from context that does not justify it, reinforcing social biases. For example:

> "My friend is a lawyer. She works late."

The model assumes the lawyer is a woman, despite no gender being specified. Such assumptions can be harmful in real-world applications.

## 🎯 Project Goals
1. **Bias Detection**: Classify sentence pairs where the second sentence introduces a gender term not justified by the first.
2. **Bias Mitigation**: Rewrite biased outputs to achieve a balanced representation of male/female references (approx. 50% each).
3. **Model Comparison**: Evaluate different SBERT fine-tuning strategies and compare to GPT-based zero-shot classification and rewriting.
4. **Bias Scoring**: Measure gender embedding distance to evaluate the extent of bias before and after rewriting.

## 📁 Dataset
- **Name**: `labeled_dataset_1`
- **Size**: ~3400 sentence pairs
- **Labeling**:
  - `1` = biased (gender inferred unjustifiably)
  - `0` = not biased
- **Generation**:
  - Used GPT-4o to generate diverse sentence pairs
  - Applied rules and dictionaries to detect gender terms in sentence 1
  - Resolved ambiguous/unisex names via replacement

## 📊 Example
```
Sentence 1: "The firefighter, soaked in sweat, emerged with a rescued puppy."
Sentence 2: "His bravery stunned the crowd."
Label: 0 (Not biased)
```

## 🧠 Models
### 1. Detection
- **GPT-3.5 / GPT-4o**: Zero-shot classification with prompt engineering.
- **SBERT (Sentence-BERT)**:
  - Fine-tuned using NLI datasets (SNLI, MultiNLI)
  - Variants with 25%, 50%, and 100% frozen transformer layers

### 2. Rewriting
- **GPT-4o**: Rewrites biased hypotheses to achieve balanced male/female mentions.

## 🧪 Evaluation
- **Classification**: Accuracy, Precision, Recall, F1-score
- **Bias Score**: Cosine distance between embeddings of rewritten sentences and reference terms "man"/"woman"
  - Goal: Biased samples → |score| >> 0, Rewritten samples → |score| ≈ 0

## 🛠️ Code Structure
```
├── data/
│   ├── Dataset.csv
│   ├── Gpt3.5_predictions.csv
│   └── Gpt4o_predictions.csv
│
├── notebooks/
│   ├── data_labeling.ipynb
│   ├── train_sbert_classifier.ipynb
│   └── gpt_prediction_loop.py
│
└── Evaluation_metrics_function_gpt.py
```

## 🥇 Results
- **Best Model**: SBERT with 25% frozen layers
  - F1 = 90.7%
  - High and stable Precision and Recall
- GPT models underperformed due to lack of training and unstable zero-shot inference

## 📍 Key Takeaways
- Combining fine-tuned SBERT for detection with GPT for rewriting offers a robust pipeline
- Modular structure allows easy adaptation for other bias domains (e.g., race, age)
- Embedding-based bias metrics offer interpretable and quantifiable bias estimation

---

## 🤝 Team
- Ariel Soffer
- Katherine Zablianov
- Shay Yeffet
- Neta Robinzon Butbul
