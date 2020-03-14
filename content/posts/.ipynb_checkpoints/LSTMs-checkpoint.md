---
title: "Recurrent Neural Networks, Lost memories, & LSTMs"
date: 2020-03-01T14:55:21+01:00
draft: true
categories:
- Review
- <2000
tags:
- RNN
- LSTM
- Vanishing Gradients
- Exploding Gradients
Description: "Memory is fascinating because it saves so much.. using so little."
---

Let's say we are interested in labelling a sequence of tokens as either positive or negative. In natural language, we may be interested in figuring out the sentiment behind a sentences (tokens = words), examples:

<div style="text-align:center">
$[\text{I},\text{Love},\text{You}] \to 1$ (Positive)<br/>
$[\text{He},\text{Killed},\text{Her}] \to 1$ (Negative)
</div><br/>

Let's denote the word-sequence $x=\{x_1,x_2,\dots,x_n\}$ and the target $y \in \text{\{0,1}\}$. For $\forall i, x_i \in \Bbb{R}^{2 \times 1}$ an embedding representing any word in our vocabulary.

To solve the task, we need to define two things:
1. Memory: What we know until now.  
2. Prediction: After traversing the sequence, we need to output a prediction.
    - Memory at the end doesn't represent the prediction.
        - Differences might be in the shape and desired range of values.

We define both memory (or hidden state $h_t$) at time $t$ and prediction $\hat{y}$ as follows:

$$\left\\{ 
\begin{array}{l}
h_t = W_h\sigma(h_{t-1})+W_xx_t + b \newline
\hat{y} = W_y \sigma(h_n)
\end{array}
\right.$$

The activation function is sigmoid $\sigma(x)=\frac{1}{1+e^{-x}}$. 

From the formulation, we realise that we have two conceptually different NNs ($[W_h, W_x], [W_y]$). 
- first learns how to create the present hidden state from past hidden state & current token. 
- second learns how to map the $n$th hidden state to the final prediction.

Finally, we need a measure of Loss or the distance between our prediction and the target. Let's simply define it as Mean Squared Error ($MSE$):

$$J(\hat{y},y)=(\hat{y}-y)^2$$

Let's visualize the computational graph for different number of steps $n$.

When $n=1$:

[GRAPH]
