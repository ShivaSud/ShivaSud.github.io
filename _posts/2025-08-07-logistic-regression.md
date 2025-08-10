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
Logistic regression is a machine learning algorithm, which is used for binary classification, for example seeing if a student passed or failed an exam, given the amount of hours they spent studying for it. The reason why logistic regression has regression in the name, is because rather than predicting the class (whether the student will pass or not), it predicts the probability of them passing given the input - how many hours they spent studying. Using this probability, a threshold value can be set up to the discretion of the programmer, to decide which class the predicted probability will fall into.

Continuing from the example above, let's say we perform an ordinary linear regression with the input being the hours the student studied on the x-axis, and then their predicted test score being on the y-axis. Now in logistic regression the aim is different as now we want the output to be a probability of the student passing rather than a predicted score.

## Multiple Features
In the previous blog post regarding linear regression, we only considered the simple case of having just one input feature, but in this blog post I will explain how we can use multiple features to predict the target variable. Initially, we considered the following line 

$$ y  = mx + c $$ 

* $y$ is the target variable 
* $m$ is the gradient 
* $x$ is the input variable (a feature)
* $c$ is the y-intercept 

Now when predicting the target variable using multiple features, we will give each feature its own value for m. This is because each feature will impact the prediction of the target variable by a different amount and so we need to train our algorithm such that it can learn these values. Using the word gradient here doesn't make sense anymore, so we will use the more accurate word **weight** as commonly used in machine learning. In the same way, the y-intercept is now called the **bias**. Hence we have the new equation:

$$ z = \theta_1x_1 + \theta_2x_2 + ... + \theta_nx_n + c $$

* $z$ is the predicted value for the target variable
* $x_n$ represents each feature
* $\theta_n$ is the weight for each feature 
* $c$ is the bias

We can write this equation more elegantly using matrix format:

$$ z = \left(\theta_{1}\space \theta_{2}\cdots \theta_{n}\right)\begin{pmatrix}x_{1}\\x_{2}\\\vdots\\x_{n}\end{pmatrix}+c $$

and then...

$$ z = \vec{\theta}^\top\vec{x}+c$$

This provides the predicted value for the $ith$ row of data, hence for clarity we can update our equation to clearly show this

$$ z^i = \vec{\theta}^\top\vec{x^i}+c$$

## The Sigmoid Function

This predicted value of y, can be any number all the way to infinity, but we need values of y between 0 and 1 since it is a probability, as mentioned earlier. The way we can accomplish this is by feeding the y into a mathematical function whose input ranges from 0 to 1 regardless of the input. Many such equations exist, however, the one that is often used for this scenario is the sigmoid function. The equation for the sigmoid function is as follows:

$$\phi(x)=\frac{1}{1+e^{-x}}$$

The graph for the sigmoid function can be seen below:
![Sigmoid function](/assets/images/Sigmoid_Function.png "Sigmoid function")

You can see why the sigmoid function is very appropriate for this scenario. It has asymptotes at 0 and 1 (you can plug in and check for yourself), large values are close to 1 and small values are close to 0, giving the function a very even spread.

Now if we input the value $y^i$ into the sigmoid function we will get a function that takes in all of the features and predicts a probability for the target value.

$$ {\hat{y}^i} = \phi(z^i)=\frac{1}{1+e^{-(\vec{\theta}^\top\vec{x^i}+c)}}$$

* ${\hat{y}^i}$ is the predicted probability of the target variable

## Creating a Likelihood function

If we want to optimise this function then we need an equation to compare the predicted probability of y, to its true label, which will be 0 or 1. We want to then train the model such that the predicted probability should be as close as possible to its label. In essence, when the true label is 0, the predicted probability should be close to 0 and when the true label is 1, the predicted probabality should be close to 1. 

Here is the mathematical equation for what I have described above:

$$\mathcal{L(\theta)} = \prod_{i = 1}^{n} \space {\hat{y}^i}^{y^i}(1-{\hat{y}^i})^{1- y^i}$$

This function essentially calculates the total error of the predictions, given the current weights. We want to optimise this function such that the value of the likelihood function is as close as possible to 1.

Let's now walk through the two possible scenarios - when the label is 1 and when the label is 0 - to better understand how the function works conceptually:
   1. If the true label is 1, then $(1-{\hat{y}^i})^{1- y^i}$ will collapse completely, because anything to the power of 0 is 1, and multiplying by 1 has no effect. Then the value of  ${\hat{y}^i}^{y^i}$ will collapse simply into ${\hat{y}^i}$ because it has been raised to the power of 1, which has no effect. Hence, we have been just left with the predicted probability. If this value is close to 1, then it won't contribute much to the error decreasing from 1. If this value is close to 0 then it will contribute a lot to the error decreasing, which is good because it is clearly predicting the wrong label.
   2. If the true label is 0, then ${\hat{y}^i}^{y^i}$ will collapse completely, because anything to the power of 0 is 1, and multiplying by 1 has no effect. Then the value of  $(1-{\hat{y}^i})^{1- y^i}$ will collapse simply into $1-{\hat{y}^i}$ because it has been raised to the power of 1, which has no effect. Hence, we have been just left with the $1 - \text{predicted probability}$. If the predicted value is close to 0, then $1 - \text{predicted probability}$ won't contribute much to the error decreasing from 1. If $1 - \text{predicted probability}$ is close to 0 then it will contribute a lot to the error decreasing, which is good because it is clearly predicting the wrong label.

## Cross-Entropy Loss function

While the above loss function works in theory, there are a few adjustments that we need to make before we can try to perform gradient descent.

First of all gradient descent works to minimise the error, however we are currently trying to maximise the error function so it can be as close to 1, which doesn't make sense in this context. To fix this we negate the loss function so that now optimising the function means that we are minimising rather than maximising the function.

$$\mathcal{L(\theta)} = -\prod_{i = 1}^{n} \space {\hat{y}^i}^{y^i}(1-{\hat{y}^i})^{1- y^i}$$

Second of all, performing multiplication on very long decimals continuously for the every single row in the dataset can be a very bad idea. This is because every time we make a multiplication, the computer will round the decimal so that it can be stored appropriately in binary. Doing this multiple times can amplify the distortion between what the actual error is, and what the computer has stored. This effect is called **numerical instability**. To prevent this we need to find a way to avoid using multiplication. The way to do this is by taking the log of the function as follows 

$$\ln\mathcal{L(\theta)} = -\sum_{i = 1}^{n} \ln(\space {\hat{y}^i}^{y^i}(1-{\hat{y}^i})^{1- y^i})$$

$$\ln\mathcal{L(\theta)} = -\sum_{i=1}^n\left[y^i\ln(\hat{y}^i)+(1-y^i)\ln(1-\hat{y}^i)\right]$$

Although, now that we are using addition we need to take the mean error from all of the samples hence we can fix our previous equation to get our final loss function:

$$\text{Cross Entropy Loss Function} = \ln\mathcal{L(\theta)} = -\frac{1}{n}\sum_{i=1}^n\left[y^i\ln(\hat{y}^i)+(1-y^i)\ln(1-\hat{y}^i)\right]$$

The equation that we have reached to is a commonly used loss function in machine learning, called the Cross Entropy Loss Function!

## Gradient Descent

Now that we have the loss function we can perform gradient descent to iteratively update the weights of each feature, so that it is tuned to correctly classify each sample into the correct categories. To be more precise, we need to see how the error changes with respect to each parameter ($\frac{\partial E}{\partial\theta_{j}}$ and $\frac{\partial E}{\partial c}$) and then minimise this error.

Currently though, we have the two functions:

$$ {\hat{y}^i} = \frac{1}{1+e^{-(\vec{\theta}^\top\vec{x^i}+c)}}$$

$$\text{E} = -\frac{1}{n}\sum_{i=1}^n\left[y^i\ln(\hat{y}^i)+(1-y^i)\ln(1-\hat{y}^i)\right]$$

To find the value of $\frac{\partial E}{\partial\theta_{j}}$ we can use the chain rule as follows:

$$\frac{\partial E}{\partial\theta_{j}} = \frac{\partial E}{\partial{\hat{y}^i}} \cdot \frac{\partial {\hat{y}^i}} {\partial\theta_{j}}$$

Let us first find $\frac{\partial E}{\partial{\hat{y}^i}}$

Since we are finding the derivative for the $ith$ value we’ll compute for a single training example $i$ and later sum over all $n$ examples.

$$\frac{\partial\mathrm{E}}{\partial\hat{y}^i}=-\frac{1}{n}\left(\frac{y^i}{\hat{y}^i}-\frac{1-y^i}{1-\hat{y}^i}\right)$$

Now let's first find $\frac{\partial {\hat{y}^i}} {\partial\theta_{j}}$: 

To break this down even further we can say:

$$ z^i = \vec{\theta}^\top\vec{x^i}+c$$

$$ {\hat{y}^i} = \frac{1}{1+e^{-z^i}}$$

$$\frac{\partial {\hat{y}^i}}{\partial\theta_{j}} = \frac{\partial {\hat{y}^i}} {\partial z^i} \cdot \frac{\partial z^i}{\partial\theta_j}$$

Let's first find $\frac{\partial {\hat{y}^i}} {\partial z^i}$

$$ {\hat{y}^i} = (1+e^{-z^i})^{-1}$$

$$\frac{\partial{\hat{y}^i}}{\partial z^i} = -1 \cdot -e^{-z^i}(1+e^{-z^i})^{-2}$$ 

$$=\frac{e^{-z^i}}{(1+e^{-z^i})^{2}}$$

$$=\frac{e^{-z^i}}{1+e^{-z^i}} \cdot \frac{1}{1+e^{-z^i}}$$

$$=\frac{(1+e^{-z^i})-1}{1+e^{-z^i}} \cdot {\hat{y}^i}$$

$$= (1-{\hat{y}^i}){\hat{y}^i}$$

Now let's find $\frac{\partial z^i}{\partial\theta_j}$

$$\frac{\partial z^i}{\partial\theta_j} = \vec{x^i_j}$$

Hence we can find $\frac{\partial {\hat{y}^i}}{\partial\theta_{j}}$

$$= (1-{\hat{y}^i}){\hat{y}^i} \cdot \vec{x^i_j}$$

And finally the value of $\frac{\partial E}{\partial\theta_{j}}$

$$ \frac{\partial E}{\partial \theta_j} = \sum_{i=1}^n (1 - \hat{y}^{i}) \hat{y}^{i} x_j^{i} \cdot \left[ -\frac{1}{n} \left( \frac{y^{i}}{\hat{y}^{i}} - \frac{1 - y^{i}}{1 - \hat{y}^{i}} \right) \right]$$

$$ = -\frac{1}{n} \sum_{i=1}^n (1 - \hat{y}^{i}) \hat{y}^{i} x_j^{i} \cdot \frac{y^{i} (1 - \hat{y}^{i}) - (1 - y^{i}) \hat{y}^{i}}{(1 - \hat{y}^{i}) \hat{y}^{i}}$$

$$ = -\frac{1}{n} \sum_{i=1}^n x_j^{i} \cdot \left( y^{i} (1 - \hat{y}^{i}) - (1 - y^{i}) \hat{y}^{i} \right)$$

$$ = -\frac{1}{n} \sum_{i=1}^n x_j^{i} \cdot \left( y^{i} - y^{i} \hat{y}^{i} - \hat{y}^{i} + y^{i} \hat{y}^{i} \right)$$

$$ = -\frac{1}{n} \sum_{i=1}^n x_j^{i} \cdot (y^{i} - \hat{y}^{i})$$


Lastly we need to find the how the error changes with respect to the bias, so we can update that too. 
$$\frac{\partial E}{\partial c_{j}} = \frac{\partial E}{\partial{\hat{y}^i}} \cdot \frac{\partial {\hat{y}^i}} {\partial c_{j}}$$

$$\frac{\partial {\hat{y}^i}}{\partial c_{j}} = \frac{\partial {\hat{y}^i}} {\partial z^i} \cdot \frac{\partial z^i}{\partial c_j}$$

$$\frac{\partial{\hat{y}^i}}{\partial z^i}= (1-{\hat{y}^i}){\hat{y}^i}$$

$$\frac{\partial z^i}{\partial c} = 1$$

$$ \frac{\partial {\hat{y}^i}}{\partial c} = (1-{\hat{y}^i}){\hat{y}^i} $$

$$ \frac{\partial E}{\partial c} = -\frac{1}{n} \sum_{i=1}^n \cdot (y^{i} - \hat{y}^{i}) $$

Now we have the two equations that we require to update the weights and bias iteratively and perform gradient descent.
# Code
``` python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
class LogisticRegression:
    
    def __init__(self):
        self.w = None
        self.c = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y, learning_rate = 0.01, epochs = 100):
        X = X.to_numpy()
        y = y.to_numpy().reshape(-1, 1)

        n_samples, n_features = X.shape
        self.w = np.zeros((n_features, 1))
        self.c = 0

        for _ in range(epochs):
            z = np.dot(X, self.w) + self.c

            y_pred = self.sigmoid(z)

            error = y - y_pred
            dw = np.dot(X.T, error) / n_samples
            dc = np.sum(error) / n_samples

            self.w += learning_rate * dw
            self.c += learning_rate * dc


    def predict(self, X):
        X = X.to_numpy()
        return self.sigmoid(np.dot(X, self.w) + self.c)

#Defining the X and y arrays

df = pd.read_csv("LogisticRegression/framingham_train.csv")
df_clean = df.dropna()
# Split features and target
X_train = df_clean.drop("TenYearCHD", axis=1)
y_train = df_clean["TenYearCHD"]

# Fit your model
model = LogisticRegression()
model.fit(X_train, y_train, learning_rate=0.0001, epochs=10000)

df = pd.read_csv("LogisticRegression/framingham_test.csv")
df_clean = df.dropna()
X_test = df_clean.drop("TenYearCHD", axis=1)
y_test = df_clean["TenYearCHD"]

# Predict
y_pred_prob = model.predict(X_test)
# Convert probabilities to binary predictions (threshold 0.5)
y_pred = (y_pred_prob >= 0.5).astype(int).flatten()
# Calculate accuracy
accuracy = (y_pred == y_test.to_numpy()).mean() * 100
print(f"Accuracy: {accuracy:.2f}%")
```
