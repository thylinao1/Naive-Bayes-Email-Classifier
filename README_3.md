# Email Spam Classifier using Naive Bayes

A probabilistic spam detection system implementing Naive Bayes classification with emphasis on mathematical foundations and numerical stability. This project demonstrates how Bayes' Theorem can be applied to binary text classification and explores the critical importance of log-space computation for production-ready machine learning systems.

## 🎯 Project Overview

This implementation builds an email spam classifier from scratch, focusing on:
- **Mathematical rigor**: Direct implementation of Bayes' Theorem and probability calculations
- **Numerical stability**: Comparison between standard and log-transform approaches
- **Real-world applicability**: Achieving production-grade precision (98%+) through proper implementation

### Key Finding

The project reveals a dramatic performance difference between two mathematically equivalent implementations:
- **Standard Naive Bayes**: 59.57% precision (40 false positives per 100 flagged emails)
- **Log-transform Naive Bayes**: 98.42% precision (2 false positives per 100 flagged emails)

This ~40 percentage point improvement stems entirely from numerical stability considerations, demonstrating that implementation details can be as important as algorithmic choice.

## 📊 Results

| Metric | Standard NB | Log-transform NB | Improvement |
|--------|-------------|------------------|-------------|
| **Accuracy** | ~85% | ~99% | +14% |
| **Recall** | 98.0% | 98.4% | +0.4% |
| **Precision** | 59.57% | 98.42% | +38.85% |
| **False Positives** | 169 | 4 | -97.6% |

## 🧮 Mathematical Foundation

### Bayes' Theorem for Classification

The classifier computes the posterior probability that an email is spam given its content:

```
P(spam | email) = [P(email | spam) × P(spam)] / P(email)
```

Where:
- **P(spam)**: Prior probability (proportion of spam in training data)
- **P(email | spam)**: Likelihood of observing this email given it's spam
- **P(email)**: Evidence (can be ignored when comparing spam vs ham)

### The Naive Independence Assumption

The "naive" aspect treats each word as independent, simplifying the likelihood calculation:

```
P(email | spam) = ∏ P(word_i | spam) for all words in email
```

While unrealistic, this assumption dramatically reduces computational complexity while maintaining strong performance.

### Laplace Smoothing

To handle words never seen in training data, we apply additive smoothing:

```
P(word | spam) = (count(word in spam) + 1) / (total spam words + |vocabulary|)
```

This ensures no probability ever becomes exactly zero, which would eliminate the email's entire probability.

### Log-Space Computation

**The Problem:** Multiplying hundreds of small probabilities causes numerical underflow in floating-point arithmetic.

**The Solution:** Transform multiplication into addition using logarithms:

```
log P(email | spam) = ∑ log P(word_i | spam)
```

Since logarithm is monotonic, comparing log probabilities produces identical classification to comparing probabilities—but with numerical stability.

## 🛠️ Implementation Details

### Preprocessing Pipeline

1. **Lowercasing**: Normalize text to reduce vocabulary size
2. **Punctuation removal**: Focus on semantic content
3. **Tokenization**: Split text into individual words
4. **Stopword removal**: Filter common words ('the', 'is', 'at', etc.)

### Classification Process

1. Build vocabulary from training data
2. Calculate prior probabilities: P(spam) and P(ham)
3. Compute word likelihoods: P(word|spam) and P(word|ham) for all words
4. For each test email:
   - Calculate log[P(email|spam) × P(spam)]
   - Calculate log[P(email|ham) × P(ham)]
   - Classify as spam if first quantity > second quantity

## 📁 Project Structure

```
naive-bayes-spam-classifier/
├── spam_classifier.ipynb    # Main implementation notebook
├── README.md               # This file
└── data/                   # (Not included - see Data Requirements)
    └── emails.csv
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install numpy pandas nltk
```

### Data Requirements

The classifier expects a CSV file with two columns:
- `text`: Email content (string)
- `spam`: Label (1 for spam, 0 for ham/legitimate)

Example format:
```csv
text,spam
"Subject: Buy now! Amazing deals...",1
"Hi John, about tomorrow's meeting...",0
```

**Note**: Due to data licensing, you'll need to provide your own email dataset. Public options include:
- [Enron Email Dataset](https://www.cs.cmu.edu/~enron/)
- [SpamAssassin Public Corpus](https://spamassassin.apache.org/old/publiccorpus/)
- [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

### Running the Classifier

1. Place your dataset as `data/emails.csv`
2. Open `spam_classifier.ipynb` in Jupyter Notebook or JupyterLab
3. Run all cells sequentially

The notebook will:
- Load and preprocess your data
- Train both standard and log-transform models
- Display comprehensive performance comparison
- Explain the mathematical reasoning behind the results

## 📈 Performance Metrics Explained

### Recall (Sensitivity)
*"Of all actual spam emails, how many did we catch?"*

Both models achieve ~98% recall, meaning they successfully identify 98 out of 100 spam emails. High recall is crucial to protect users from spam.

### Precision (Positive Predictive Value)
*"Of all emails we flagged as spam, how many were actually spam?"*

This is where the models diverge dramatically:
- Standard model: 59.57% (41 out of 100 flagged emails are actually legitimate)
- Log model: 98.42% (only 2 out of 100 flagged emails are legitimate)

High precision is essential to avoid filtering important emails to the spam folder.

### Why This Matters

A spam filter with low precision frustrates users by hiding legitimate emails. The log-transform model's 98.42% precision makes it production-ready, while the standard implementation's 59.57% would be unacceptable in practice.

## 🔬 Technical Insights

### Numerical Underflow Demonstration

Consider an email with 100 words, each with P(word|spam) ≈ 0.01:

**Standard approach:**
```
P(email|spam) = 0.01^100 ≈ 10^-200
```
This underflows to exactly 0.0 in double-precision floating point.

**Log approach:**
```
log P(email|spam) = 100 × log(0.01) = 100 × (-4.605) = -460.5
```
This remains numerically tractable.

### When Underflow Destroys Classification

Once standard Naive Bayes probabilities underflow to zero:
1. Both P(spam|email) and P(ham|email) become 0
2. The model cannot distinguish between classes
3. Classification becomes arbitrary/random
4. Precision collapses

The log-transform elegantly sidesteps this issue entirely.

## 🎓 Learning Outcomes

This project demonstrates:

1. **Theoretical foundations**: How probability theory translates to practical ML algorithms
2. **Implementation matters**: Two mathematically equivalent formulations yielding vastly different results
3. **Numerical considerations**: Why log-space computation is standard practice in probabilistic models
4. **Evaluation nuance**: Understanding multiple metrics (accuracy, precision, recall) and their trade-offs
5. **Production readiness**: The gap between "working" code and production-quality implementations

## 🔮 Future Enhancements

Potential extensions to explore:
- **Bigrams/trigrams**: Consider word pairs/triples instead of individual words
- **TF-IDF weighting**: Give more importance to rare, informative words
- **Feature engineering**: Email metadata (sender, time, length, etc.)
- **Threshold tuning**: Adjust decision boundary for precision-recall trade-offs
- **Cross-validation**: More robust performance estimation
- **Comparison with other models**: Logistic regression, SVM, neural networks

## 📚 References

This implementation was built from scratch to understand the mathematical foundations of Naive Bayes classification and the critical importance of numerical stability in probabilistic machine learning.

