# Gaussian Discriminant Analysis (GDA)

## Overview

Gaussian Discriminant Analysis is a **generative learning algorithm** used for classification. Unlike discriminative models (e.g., logistic regression), which directly model \( P(y \mid x) \), GDA models the **joint probability distribution** \( P(x, y) \) by first estimating \( P(x \mid y) \) and \( P(y) \), and then applying Bayes' rule.

GDA assumes that the features \( x \in \mathbb{R}^n \) conditioned on the class label \( y \in \{0, 1\} \) are distributed according to a **multivariate Gaussian**.

## Model Assumptions

1. \( y \sim \text{Bernoulli}(\phi) \)
2. \( x \mid y = 0 \sim \mathcal{N}(\mu_0, \Sigma) \)
3. \( x \mid y = 1 \sim \mathcal{N}(\mu_1, \Sigma) \)

Here:
- \( \phi \) is the prior probability \( P(y = 1) \)
- \( \mu_0, \mu_1 \in \mathbb{R}^n \) are the class means
- \( \Sigma \in \mathbb{R}^{n \times n} \) is the **shared** covariance matrix across classes

## Parameter Estimation (MLE)

Given a labeled dataset \( \{(x^{(i)}, y^{(i)})\}_{i=1}^m \), the maximum likelihood estimates are:

- \( \phi = \frac{1}{m} \sum_{i=1}^m 1\{y^{(i)} = 1\} \)
- \( \mu_0 = \frac{\sum_{i=1}^m 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 0\}} \)
- \( \mu_1 = \frac{\sum_{i=1}^m 1\{y^{(i)} = 1\} x^{(i)}}{\sum_{i=1}^m 1\{y^{(i)} = 1\}} \)
- \( \Sigma = \frac{1}{m} \sum_{i=1}^m (x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T \)

## Prediction

To predict a label for a new example \( x \), use Bayes’ Rule:

\[
P(y = 1 \mid x) = \frac{P(x \mid y=1)P(y=1)}{P(x)}
\]

The predicted class is:

\[
\hat{y} = \arg\max_{y \in \{0, 1\}} \; P(x \mid y)P(y)
\]

Because \( P(x \mid y) \) is Gaussian, this results in a **linear decision boundary** if \( \Sigma \) is shared, and a **quadratic boundary** if each class has its own \( \Sigma_y \).

## Comparison with Logistic Regression

- **GDA** is generative: it models \( P(x \mid y) \), then uses Bayes’ rule.
- **Logistic regression** is discriminative: it models \( P(y \mid x) \) directly.
- If the Gaussian assumption holds, **GDA converges faster** with fewer training examples.
- If the assumption is violated, **logistic regression is more robust**.

## Advantages

- Closed-form parameter estimates via MLE
- Efficient for low-dimensional problems
- Fast inference due to analytic form of posterior

## Limitations

- Assumes Gaussianity of features within each class
- Sensitive to feature correlations
- Requires invertibility of the covariance matrix

## References

- [CS229 Lecture Notes (Andrew Ng) – Chapter 4](https://cs229.stanford.edu/notes2022fall/cs229-notes4.pdf)
- Murphy, K. P. *Machine Learning: A Probabilistic Perspective*. Chapter 4.
- Bishop, C. M. *Pattern Recognition and Machine Learning*

---

