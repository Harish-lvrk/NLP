# Quora Question Pairs Semantic Similarity

This project determines if two questions are semantically similar (duplicate intent) using **Natural Language Processing (NLP)** techniques and a machine learning pipeline. The trained model is deployed as an interactive web application using **Streamlit**.

---

## 📜 Table of Contents

* [Project Overview](#project-overview)
* [Dataset](#dataset)
* [Technical Stack](#technical-stack)
* [Features](#features)
* [Model and Evaluation](#model-and-evaluation)
* [Installation and Usage](#installation-and-usage)
* [File Structure](#file-structure)
* [Acknowledgements](#acknowledgements)

---

## 📝 Project Overview

This project tackles the **Quora Question Pairs** NLP problem. Given a pair of questions, the task is to predict whether they are **duplicates** (semantically identical) or not.

**Key contributions:**

* Built a **preprocessing pipeline** for text cleaning, tokenization, and feature extraction.
* Engineered **15+ numerical features** capturing different aspects of text similarity.
* Trained a **Random Forest classifier** achieving strong performance (F1-score: 0.82, Accuracy: 0.84).
* Developed an **interactive Streamlit app** to provide real-time predictions.

---

## 📊 Dataset

The dataset is from **Quora Question Pairs**, containing over 400,000 question pairs labeled as duplicates or not.

* [Quora Question Pairs Dataset on Kaggle](https://www.kaggle.com/c/quora-question-pairs)

---

## 🛠 Technical Stack

* **Programming Language:** Python 3.x
* **Data Processing:** pandas, numpy
* **NLP & Feature Engineering:** fuzzywuzzy, scikit-learn (CountVectorizer, TfidfVectorizer)
* **Machine Learning:** scikit-learn (Random Forest, Logistic Regression)
* **Visualization:** matplotlib, seaborn, plotly
* **Deployment:** Streamlit

---

## ✨ Features

### 1. Basic Features

| Feature                        | Description                              |
| ------------------------------ | ---------------------------------------- |
| `q1_len`, `q2_len`             | Character lengths of questions           |
| `q1_num_words`, `q2_num_words` | Number of words in each question         |
| `common_words`                 | Count of common words between questions  |
| `total_words`                  | Total unique words across both questions |
| `word_share`                   | Ratio of common words to total words     |

### 2. Token-Based Features

* Ratios of common words, stopwords, and tokens
* Checks for first and last word equality between questions

### 3. Length-Based Features

| Feature                | Description                                                         |
| ---------------------- | ------------------------------------------------------------------- |
| `abs_len_diff`         | Absolute difference in word counts                                  |
| `mean_len`             | Average length of both questions                                    |
| `longest_substr_ratio` | Ratio of longest common substring length to minimum question length |

### 4. Fuzzy Features (using `fuzzywuzzy`)

| Feature              | Description                                              |
| -------------------- | -------------------------------------------------------- |
| `fuzz_ratio`         | Basic similarity score                                   |
| `fuzz_partial_ratio` | Best matching substring similarity                       |
| `token_sort_ratio`   | Similarity after alphabetically sorting tokens           |
| `token_set_ratio`    | Similarity considering common tokens regardless of order |

### 5. Bag of Words (BoW)

* Used `CountVectorizer` to convert text into sparse numeric vectors for model input.

---

## 🧠 Model and Evaluation

* **Model:** Random Forest Classifier
* **Features Used:** Engineered features + BoW vectors
* **Performance:**

  * Accuracy: 0.84
  * F1-score: 0.82

**Methodology:**

1. Preprocessing and cleaning the text
2. Engineering multiple similarity-based features
3. Splitting the dataset into training and validation sets
4. Training Random Forest and evaluating performance metrics
5. Deploying model with Streamlit

**Example Input/Output:**

```
Input: ('How do I learn Python?', 'What is the best way to learn Python?')
Output: Duplicate (Probability: 0.91)
```

---

## 🚀 Installation and Usage

1. **Clone the repository:**

```bash
git clone https://github.com/your-username/quora-question-pairs.git
cd quora-question-pairs
```

2. **Create and activate a virtual environment:**

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app:**

```bash
streamlit run streamlit-app/app.py
```

---

## 📁 File Structure

```
quora-question-pairs/
│
├── streamlit-app/
│   ├── app.py             # Main Streamlit application script
│   ├── helper.py          # Feature engineering and preprocessing functions
│   ├── model.pkl          # Trained machine learning model
│   ├── cv.pkl             # Trained CountVectorizer
│   └── requirements.txt   # Python dependencies
│
├── notebooks/             # Jupyter notebooks for EDA and modeling
├── data/                  # Raw and processed dataset files
└── README.md              # Project documentation
```

---

## 🙏 Acknowledgements

* **Conceptual Learning:** Springboard NLP lectures, YouTube tutorials on NLP, feature engineering, and deployment.
* **AI Assistance:** Gemini Pro and ChatGPT for code generation and optimization.
* **Community:** Kaggle for datasets, Stack Overflow for technical guidance.
