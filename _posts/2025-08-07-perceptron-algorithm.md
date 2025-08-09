---
title: Logistic Regression
date: 2025-08-07
categories: [Machine Learning]
tags: [machine-learning]
math: true
comments: true
---

# Theory

## Overview

The perceptron algorithm, is a learning algorithm that separates two classes of data with a line such that it can classify a new piece of data into a class accordingly. Essentially, if we have the same equation as in the previous blog:

$$ z = \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n + c $$

$$ z = \vec{\theta}^\top\vec{x_i}+c$$

then if after plugging in the input features, we get a value that is greater than zero, then we know that the data point is on one side of the hyperplane (we say this for n dimensional boundaries), and hence can identify which class it is in. If, after plugging in the input features, the value of $z$ is less than zero then we can infer that the point is on the other side of the hyperplane, and hence we can identify which class the value is in accordingly. 

## Loss Function 

Now we need a loss function to adjust the value of the weights such that it creates the most optimal line to separate the two classes. In the perceptron algorithm, the loss function works slightly differently compared to the previous two algorithms as we only input the points that were misclassified in the previous iteration into the loss function. 

Given a label $y_i\in\{-1,+1\}$, the perceptron makes a mistake when:

$$y_iz_i\leq0$$

Essentially, when the label and the prediction disagree, their product becomes negative, indicating that the perceptron has made a mistake.

Since we only want to use the misclassified points in our loss function, here is the actual equation:

$$L(\theta)=\sum_{i = 1}^n\max(0,-y_iz_i)$$

If the point has been misclassified then $y_iz_i$, will be negative and so $-y_iz_i$, and therefore will be greater than zero so we can add it to the error.

## Gradient Descent 

Now that we have a loss function, we can find the partial derivatives of the error with respect to the weights/the bias and use this to adjust their values to become more optimised. We can rewrite the equation as such:

$$ L = \sum_{i = 1}^n\max(0,-y_i \cdot(\vec{\theta}^\top\vec{x_i}+c))$$

$$\frac{\partial L}{\partial\theta}=\sum_{i=1}^n\begin{cases}-y_ix_i,&\mathrm{if}y_i(\theta^\top x_i+c)\leq0\\0,&\mathrm{otherwise}&\end{cases}$$

$$ \frac{\partial L}{\partial c}=\sum_{i=1}^n\begin{cases}-y_i,&\mathrm{if~}y_i(\theta^\top x_i+c)\leq0\\0,&\mathrm{otherwise}&\end{cases}$$

We can now use these equations to optimise the weights and the bias
## Code

``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Perceptron:
    def __init__(self):
        self.w = None
        self.c = None

    def fit(self, X, y, learning_rate = 0.01, epochs = 1000):
        X = X.to_numpy()
        y = y.to_numpy()

        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.c = 0

        for _ in range(epochs):
            dw = np.zeros(n_features)
            dc = 0
            for i in range(n_samples):
                z = np.dot(X[i], self.w) + self.c
                if y[i] * z <= 0:
                    dw += y[i] * X[i]
                    dc += y[i]
            self.w += dw * learning_rate
            self.c += dc * learning_rate

    def predict(self, X):
        X = X.to_numpy()
        return np.dot(X, self.w) + self.c > 0

df = pd.read_csv("PerceptronAlgorithm/perceptron_dataset.csv")
X = df.drop("label", axis = 1)
y = df["label"]

model = Perceptron()
model.fit(X, y)

# Plotting
plt.figure(figsize=(8,6))
plt.scatter(X[y == 1].iloc[:, 0], X[y == 1].iloc[:, 1], color='green', label='Label 1')
plt.scatter(X[y == -1].iloc[:, 0], X[y == -1].iloc[:, 1], color='blue', label='Label -1')

if model.w is not None and model.w.shape[0] == 2:
    x1_vals = np.linspace(X.iloc[:, 0].min(), X.iloc[:, 0].max(), 100)
    w1 = float(model.w[0])
    w2 = float(model.w[1])
    c_val = float(model.c)
    x2_vals = -(w1 * x1_vals + c_val) / w2
    plt.plot(x1_vals, x2_vals, color='red', label='Decision Boundary')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.title('Perceptron Classification')
plt.show()
```
Here is the line that classifies two sets of data using our code:
![Perceptron Classification](/assets/images/Perceptron_Classification.png "Perceptron Classification")
