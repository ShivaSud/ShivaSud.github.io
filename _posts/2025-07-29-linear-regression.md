---
title: Linear Regression
date: 2025-07-29
categories: [Machine Learning]
tags: [machine-learning]
math: true
comments: true
---

This is the first blog of a long series, where I will explain how common machine learning algorithms work intuitively from scratch. I hope you enjoy!

# Theory

Linear regression is a type of supervised learning algorithm, in which a line of best fit is calculated based on a set of bivariate data during the training process. This line of best fit is then used to make predictions of one of the variables using the other variable.

The equation of a straight line is:

$$ y= mx + c$$

where x and y are the variables that are used to train the values of m (the gradient) and c (the y-intercept). Initially we set these values to set these two values to zero, and we want to find the two values of these variables that best matches the data.

In each iteration we need to find how badly our line matches the given data, and then adjust this so that it better matches the data. This is called a **loss function**.

![Scattered data](/assets/images/Scattered_data.jpeg "Scattered data")

If we imagine the line of best fit, then we can infer that the best line
is when the line is the closest to all of the points in the scatter graph. An easy way to mathematically calculate this is by finding the average vertical distance from each point to the line. This is a commonly used loss function called the **mean squared error**. The reason that we square the data is because we care about the distance from the point to the line, and not its displacement, hence by squaring the values, everything is positive. 

Here I have shown the equation for the root mean squared error:

$$ E: \frac{1}{n} \sum_{i=1}^{n} (y_{i} - \hat{y}_{i})^{2}$$

* $n$ is the number of data points
* $y_i$ represents the observed values
* $\hat{y}_{i}$ represents the predicted values

Now we have the loss function we need to figure out how we can use the result to actually change the values of m and c. Since we want to minimise the loss function we can use calculus and differentiate the loss function with respect to m and c and use this value to adjust their values in the correct direction. 

To do this we need to write the loss function in terms of m and c. This gives us:

$$E: \frac{1}{n} \sum_{i=1}^{n} (y_{i} - (mx_i +c))^{2}$$

First, lets find how the error changes with respect to the gradient by finding the partial derivative. We can use the chain rule and find the derivative of what is inside the function, giving us $x_{i}$ and then take one from the power of the outer function multiply by 2, and multiply the two together to get the following.

$$\frac{\partial E}{\partial m}=\frac{1}{n}\cdot\sum_{i=0}^{n}2\cdot(y_{i}-(m\cdot x_{i}+c))\cdot(x_{i})$$

Now we can repeat this process by finding how the error changes with respect y-intercept, and get the following.

$$\frac{\partial E}{\partial c}=\frac{1}{n}\cdot\sum_{i=0}^{n}2\cdot (y_{i}-(m\cdot x_{i}+c))$$

We can use these derivatives now to adjust the value m and c. If we subtract the value of $\frac{\partial E}{\partial m}$ from the original value of the gradient then the next time round the change in the error with respect to the gradient will decrease, up until the point at which this error is zero and the gradient of the line matches the data, as closely as possible. The same logic works for the y-intercept, eventually providing us with the line of best fit for the provided data.

Sometimes, the value for the partial derivative can be so large that when we subtract it from the original value of m or c, we overshoot and miss the point where the error is minimum, hence failing the task. To prevent this we multiply the derivative by a **learning rate**, which is normally a small decimal that scales the partial derivative down, hence stopping the value of m and c from overshooting from what is optimal.

We also need to tell the algorithm when to stop. We code wait for the change in error to be within an acceptable range and then stop, or we could have a fixed amount of iterations called an **epoch**. In the code below I have used epochs.
# Code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    #Initialising m and c
    def __init__(self):
        self.m = 0
        self.c = 0
    
    #Training
    def fit(self, X, y, learning_rate = 0.00000001, epochs = 150):
        m = 0
        c = 0

        for i in range(epochs):
            m_partial_derivative = 0
            for i in range(len(df)):
                m_partial_derivative += X.iloc[i] * ((y.iloc[i]) - (m * X.iloc[i] + c))
            m_partial_derivative *= -2 / len(df)

            c_partial_derivative = 0            
            for i in range(len(df)):
                c_partial_derivative += (y.iloc[i]) - (m * X.iloc[i] + c)
            c_partial_derivative *= -2 / len(df)

            m = m - (learning_rate * m_partial_derivative) 
            c = c - (learning_rate * c_partial_derivative)

        self.m = m
        self.c = c 
    
    #Prediction
    def predict(self, X):
        return self.m * X + self.c

#Defining the X and y arrays
df = pd.read_csv("data.csv")
X = df["X"]
y = df["Y"]

model = LinearRegression()
model.fit(X, y)

#Plotting the data
plt.scatter(X, y, color="blue", label="Data")
x_vals = np.array([X.min(), X.max()])
y_vals = model.predict(x_vals)
plt.plot(x_vals, y_vals, color="red", label="Regression Line")
plt.legend()
plt.show()
```
Here is the trained line of best fit!
![Trained Line](/assets/images/Trained_Line.png "Trained Line")

