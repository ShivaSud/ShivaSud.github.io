---
title: K Nearest Neighbours
date: 2025-08-09
categories: [Machine Learning]
tags: [machine-learning]
math: true
comments: true
---

# Theory

## Overview

In this blog, I will be explaining the K Nearest Neighbours algorithm. This is a classification algorithm that can predict which class a new point belongs to. 

The way it works is by finding the Euclidean distance from the new point to each other points in the dataset. Then after taking the k closest points, you need to rank all of the labels of those points. The label that is most frequent in those k points is the one that is predicted for the new point. 
