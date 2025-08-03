---
title: Logistic Regression
date: 2025-07-29
categories: [Machine Learning]
tags: [machine-learning]
math: true
comments: true
---

# Theory

Logistic regression is a machine learning algorithm, which is used for binary classification, for example seeing if a student passed or failed an exam, given the amount of hours they spent studying for it. The reason why logistic regression has regression in the name, is because rather than predicting the class (whether the student will pass or not), it predicts the probability of them passing given the input - how many hours they spent studying. Using this probability, a threshold value can be set up to the discretion of the programmer, to decide which class the predicted probability will fall into.

Continuing from the example above, lets say we perform an ordinary linear regression with the input being the hours the student studied on the x-axis, and then their predicted test score being on the y-axis. The difference now, when using logistic regression is that the values on the y-axis can only range from 0 to 1, since we are dealing with a probability, whereas before the y-axis was not constrained to a fixed range.

# Code
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    def __init__(self):
        pass

    def fit():
        pass
    
    def predict():
        pass
```
