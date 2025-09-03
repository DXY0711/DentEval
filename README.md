# DentEval: Fine-tuning-Free Expert-Aligned Assessment in Dental Education via LLM Agents

DentEval is an automated assessment framework tailored for dental education. It leverages **Large Language Models (LLMs)** with **Retrieval-Augmented Generation (RAG)** and **multi-agent evaluation strategies** to deliver **fine-tuning-free**, expert-aligned scoring of student answers. The framework incorporates **rubric-based grading**, **few-shot prompting**, and **adaptive sample answer generation** to ensure consistency with human grading standards.

---

## ✅ Key Features

- **Fine-Tuning-Free Design**  
  Uses API-based LLMs without additional training.
  
- **Multi-Agent Role-Playing**  
  Simulates professors and evaluators for unbiased assessment.
  
- **Adaptive RAG**  
  Dynamically retrieves relevant content to enhance answer evaluation.
  
- **Rubric-Based Grading**  
  Supports detailed rubrics (0–5 scale) for structured scoring.
  
- **Few-Shot and Sample Answer Integration**  
  Improves alignment through exemplar answers.
  
- **Performance Metrics**  
  Computes **Accuracy**, **Spearman Rank Correlation (SROCC)**, and **Pearson Correlation (PLCC)**.
  
- **Bias Mitigation**  
  Implements **discrete histogram matching (CDF mapping)** to reduce prediction bias.

---

## 📂 Project Structure

```
DentEval/
│
├── main.py                         # Main script for evaluation
├── assessment.py                   # Core assessment logic
├── adaptive_RAG.py                 # Retrieval-Augmented Generation module
├── advanced_text_inference.py      # Text inference engine
├── requirements.txt                # Python dependencies
└── README.md                       # Documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/DXY0711/DentEval.git
cd DentEval
```

2. **Create virtual environment (recommended)**

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## 📂 Data Format

Student answer datasets should be stored in JSON files under a directory, for example:

```
{
  "questions": [
    {
      "question_id": "Q1",
      "student_answer": "Composite resin placement in layers...",
      "score": 4,
      "level": "Intermediate"
    },
    ...
  ]
}
```

**Important:** The folder path for student answers should be updated in `main.py` (see `folder_path` variable).

---

## 🚀 Quick Start

First time, you should create a knowledge base through Milvus
These files could help you:
```bash
file embedding_model
batch_processor.py
save.py
store.py
```


### **Run Example Evaluation**

```bash
python main.py
```

This will:

✅ Load a sample dental question and rubric  
✅ Extract student answers from the dataset folder  
✅ Generate **few-shot examples** for enhanced evaluation  
✅ Perform **hyperparameter search** (SA, k, few-shot)  
✅ Compute **Accuracy, SROCC, PLCC**  
✅ Save outputs:

- `few_shot_examples_q1.txt` (selected examples for few-shot prompting)
- `best_results_summary_q1.json` (final predictions and performance metrics)

---

## ⚙️ Parameters in `main.py`

- **`q1_query`**: Clinical question text  
- **`rubric`**: Rubric description  
- **`folder_path`**: Path to student answer JSON files  
- **`k`**: Maximum number of sample answers for SA  
- **`few_shot_examples`**: Generated few-shot context  

Modify these for your own dataset and questions.

---

## 📈 Evaluation Metrics

- **Accuracy**: Proportion of correct predictions  
- **SROCC**: Spearman Rank Correlation for ordinal consistency  
- **PLCC**: Pearson Correlation for linear agreement  

---

## 📚 Citation

If you use this code in your research, please cite:

```
@article{deng2025denteval,
  title={DentEval: Fine-tuning-Free Expert-Aligned Assessment in Dental Education via LLM Agents},
  author={Xinyu Deng, Vesna Miletic, Elvis Trinh, Jinlong Gao, Chang Xu, and Daochang Liu},
  journal={MICCAI},
  year={2025}
}
```

