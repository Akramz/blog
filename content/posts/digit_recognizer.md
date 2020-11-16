---
title: "Backpropagation Applied to Handwritten Zip Code Recognition (1989)"
date: 2020-01-10T14:55:21+01:00
draft: false
categories:
  - Paper Implementation
  - Review
tags:
  - Machine Learning
  - Deep Learning
  - backpropagation
  - CNN
Description: "A Review, Discussion, and Implementation of a foundational deep learning paper."
---

![Backpropagation Applied to Handwritten Zip Code Recognition](https://i.imgur.com/2IAUNy9.png)

In this article, we will go over the highly influential paper that first combined **backpropagation** and **Convolutional Neural Networks** and used end-to-end learning to recognize digits in images, titled: "[Backpropagation Applied to Handwritten Zip Code Recognition (1989)](https://www.ics.uci.edu/~welling/teaching/273ASpring09/lecun-89e.pdf)", currently having `>5750` citations in [google scholar](https://scholar.google.com/scholar?hl=en&as_sdt=0%2C5&q=Backpropagation+Applied+to+Handwritten+Zip+Code+Recognition&btnG=).

In this article, we'll go over the paper, reviewing and discussing its content while soft-implementing it line-by-line. I will use [`PyTorch`](https://pytorch.org/) to build the basic network and [`PyTorch-Lightning`](https://github.com/williamFalcon/pytorch-lightning) to refactor the code into one class. This particular model doesn't have a massive number of parameters, so we can train it on the CPU. You will find the content of the paper in quotes and my notes in normal text, and **to train the model yourself: [Source Code](https://github.com/Akramz/digit_rec)**.

#### Table of Contents

0. [Abstract](#abstract)
1. [Introduction](#introduction)
2. [Zip Codes](#zip-codes)  
   2.1. [Data Base](#data-base)  
   2.2. [Preprocessing](#preprocessing)
3. [Network Design](#network-design)  
   3.1. [Input & Output](#input-&-output)  
   3.2. [Feature Maps & Weight Sharing](#feature-maps-&-weight-sharing)  
   3.3. [Network Architecture](#network-architecture)
4. [Experimental Environment](#experimental-environemnt)
5. [Results](#results)  
   5.1. [Comparison with Other Work](#comparison)  
   5.2. [DSP Implementation](#dsp-implementation)
6. [Conclusion](#conclusion)

# Abstract

> The ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain. This paper demonstrates how such constraints can be integrated into a backpropagation network through the architecture of the network. This approach has been successfully applied to the recognition of handwritten zip code digits provided by the U.S. Postal Service. A single network learns the entire recognition operation, going from the normalized image of the character to the final classification.

Fully-connected networks are considered "universal approximators" ([Zhou Lu 2017](http://papers.nips.cc/paper/7203-the-expressive-power-of-neural-networks-a-view-from-the-width.pdf)), meaning given enough resources (parameters, compute, and training data), they can approximate any function. However, in practical situations, training a fully-connected network comes at a huge cost in terms of the required data and computational power that is needed to ensure generalizability.

The authors argue that _the ability of learning networks to generalize can be greatly enhanced by providing constraints from the task domain_, here the task domain is all that's **visual**. Generally, if we want to train a model that does not [overfit](https://en.wikipedia.org/wiki/Overfitting) on the training set and generalizes well on new data points, we need to **constrain** it in one way or many, we can constrain it through its architecture, by providing it with more data, and by using regularization techniques, the authors opted to constrain the network through its architecture by using **weight sharing**.

To understand the trade-off between data and parameters, let's consider the following example: say we have many training points $(x_0, y_0), (x_1, y_1), \dots, (x_m, y_m) \in \Bbb{R}^2$ where $y_i \sim \mathscr{N}(\mathbf{x_{i}}\Theta, \sigma)$ (true distribution) and we want to learn a function $f$ that maps $X \to y$, We consider $f$ to be a polynomial $f(x)=\theta_{0}+\theta_{1}x+\theta_{2}x^2+\dots+\theta_{n}x^{n}$ with $\theta=[\theta_0,\dots,\theta_n]^{T}$ to be learned and $n \in \Bbb{N}$ to be chosen.

Choosing a large `n` will give our model more predictive power. What will happen though is with such power the model will easily fit all of the training points to minimize the loss function but will do terribly on the test set (**overfitting**), the solution to overfitting is to somehow **constrain** the model by either giving it more data or reducing its complexity or freedom, meaning we reduce `n`:

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/dU2Hj1F.png" style="border:0;">
  <figcaption style="font-size:12px; color:'#484848'">Taken & Extended from Andrew Ng's <a target="_blank" href="https://www.coursera.org/lecture/machine-learning/the-problem-of-overfitting-ACpTQ">course</a></figcaption>
</figure>

The idea of having data constrain our model looks good (figure: bottom right) because we don't have to worry about `n`, but in practical situations (especially with big models), we won't have enough data. We usually opt for constraining the model by minimizing the number of parameters (or using regularizers) but not to the point of underfitting the data (top/bottom left side). In the paper, the authors minimized the number of free parameters by using **weight sharing**.

On top of constraining the network, the authors trained it using **backpropagation** ([Rumelhart et al 1986](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)) or by calculating the gradients of the loss function with respect to network parameters using the [chain rule](https://en.wikipedia.org/wiki/Chain_rule). Weight sharing and Backpropagation were used to solve the task of handwritten zip code digit recognition. The authors also described an end-to-end learning network that takes in the preprocessed digit images and predicts the digit without any feature engineering/extraction. Next, we will go over each concept in more detail.

# Introduction

> Previous work performed on recognizing simple digit images ([LeCun 1989](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.479&rep=rep1&type=pdf)) showed that good generalization on complex tasks can be obtained by designing a network architecture that contains a certain amount of a priori knowledge about the task. The basic design principle is to reduce the number of free parameters in the network as much as possible without overly reducing its computational power. Application of this principle increases the probability of correct generalization because it results in a specialized network architecture that has a reduced entropy ([Denker et al 1987](https://pdfs.semanticscholar.org/33fd/c91c520b54e097f5e09fae1cfc94793fbfcf.pdf), [S. Patarnello and P. Carnevali 1987](https://iopscience.iop.org/article/10.1209/0295-5075/4/4/020), [Tishby et al 1989](https://ieeexplore.ieee.org/abstract/document/118274/), [LeCun 1989](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.479&rep=rep1&type=pdf)), and a reduced Vapnik-Chervonenkis dimensionality ([Baum and Haussler 1989](http://papers.nips.cc/paper/154-what-size-net-gives-valid-generalization.pdf)).

To ensure good performance out of any neural network, we need to find a good balance between constrainability (minimal free parameters) and computational power (maximum connections). So in this paper, the authors propose a network that is well suited for visual tasks, meaning, designing an architecture that contains a certain amount of prior task knowledge within the choice of its architecture.

> In this paper, we apply the backpropagation algorithm ([Rumelhart et al 1986](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf)) to a real-world problem in recognizing handwritten digits taken from the U.S Mail. Unlike previous results reported by our group on this problem ([Denker et al 1989](http://papers.nips.cc/paper/107-neural-network-recognizer-for-hand-written-zip-code-digits.pdf)), the learning network is directly fed with images, rather than feature vectors, thus demonstrating the ability of backpropagation networks to deal with large amounts of low-level information.

# Zip Codes

### Data Base

<div style="text-align:center"><img style="width:66%;" src="http://i.imgur.com/240u0jB.png" /></div>

> The database used to train and test the network consists of `9,298` segmented numerals digitized from handwritten zip codes that appeared on U.S. mail passing through the Buffalo, NY post office. Examples of such images are shown in figure 1. The digits were written by many different people, using a great variety of sizes, writing styles, and instruments, with widely varying amounts of care, `7,219` examples are used for training the network and `2,007` are used for testing the generalization performance. One important feature of this database is that both the training set and the testing set contain numerous examples that are ambiguous, unclassifiable, or even misclassified.

The authors note the diversity of the handwritten digits, `77.7%` of the data set was used for training the network and the remaining `22.3%` was used for testing.

### Preprocessing

> Locating the zip code on the envelope and separating each digit from its neighbors, a very hard task in itself, was performed by Postal Service contractors ([Wang and Srihari 1988](https://link.springer.com/article/10.1007/BF00133697)). At this point, the size of a digit image varies but is typically around `40x60` pixels. A linear transformation is then applied to make the image fit in a `16x16` pixel image. This transformation preserves the aspect ratio of the character, and is performed after extraneous marks in the image have been removed. Because of the linear transformation, the resulting image is not binary but has multiple gray levels, since a variable number of pixels in the original image can fall into a given pixel in the target image. The gray levels of each image are scaled and translated to fall within the range `[-1,1]`.

The manual work that has been done to extract individual digits from documents is presently offered as services by many companies, the way [ImageNet](researchgate.net/profile/Li_Jia_Li/publication/221361415_ImageNet_a_Large-Scale_Hierarchical_Image_Database/links/00b495388120dbc339000000/ImageNet-a-Large-Scale-Hierarchical-Image-Database.pdf) used [Amazon Mechanical Turk](https://www.mturk.com/) for image labeling is a famous use case.

Unfortunately, the data set is not available, so we opt to use a different but highly similar and standardized data set: [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology), which contains `60,000` training images and `10,000` testing images of size `28x28` pixels. We will only use the training set, to get as close as possible to the paper's database, we will do the following:

1. Randomly sample `9,298` image indices.
2. Collect & bilinearly resize each collected image to `16x16` pixels.
3. Split the data set into training and testing (paper's ratios).
4. Translate pixel values to fall into `[-1,1]`.
5. Provide two `PyTorch` [DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)s for both training/validation datasets.

Necessary imports:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
import numpy as np
import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
```

A `PyTorch` [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset) class to contain the MNIST data.

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
class Digits(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return len(self.X)
```

A function to get the dataloaders:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
def get_digits(img_size=16, batch_size=1):
    """Gets MNIST's dataloaders.

    # Arguments
        img_size: int, resize value.
        batch_size: int, batch size for PyTorch's `DataLoader`.
    """

    # 1
    X, y = list(), list()
    indices = np.random.permutation(60000)[:9298]

    # 2
    trfs = transforms.Compose([transforms.Resize(img_size)])
    mnist = datasets.MNIST('~/data/', transform=trfs, download=True)

    for idx in indices:
        x_i, y_i = mnist[idx]
        X.append(x_i[None, ...])
        y.append(y_i)

    # 3
    X, y = torch.cat(X), torch.Tensor(y)
    X_train, y_train, X_val, y_val = X[:7291], y[:7291], X[7291:], y[7291:]

    # 4
    X_train = (2 * X_train) - 1
    X_val = (2 * X_val) - 1

    # 5
    train_ds, val_ds = Digits(X_train, y_train), Digits(X_val, y_val)
    train_ld = DataLoader(train_ds, batch_size=batch_size)
    val_ld = DataLoader(val_ds, batch_size=batch_size)
    return train_ld, val_ld
```

# Network Design

### Input & Ouput

> The remainder of the recognition is entirely performed by a multilayer network. All of the connections in the network are adaptive, although heavily constrained, and are trained using backpropagation. This is in contrast with earlier work ([Denker et al 1989](http://papers.nips.cc/paper/107-neural-network-recognizer-for-hand-written-zip-code-digits.pdf)) where the first few layers of connections were hand-chosen constants implemented on a neural network chip. The input of the network is a `16x16` normalized image. The output is composed of `10` units (one per class) and uses place coding.

What is referred to here as _"place coding"_ is what we commonly refer to as one-hot encoding, where each $y^{(i)}$ is in the form of a fixed-length vector filled with zeros except a `1` in the element with an index pointing to the target. In the context of our problem, we are classifying images of digits ranging from `0` to `9` (10 digits), so the targets get transformed like this:

- `0` $\to$ `[1,0,0,...,0]`
- `1` $\to$ `[0,1,0,...,0]`
- ...
- `9` $\to$ `[0,0,0,...,1]`

We give the network an image of size `16x16` ($X^{(i)} \in [-1,1]^{1 \times 1 \times 16 \times 16}$, the first `1x1` in the dimensions refer to `batch_size=1` and image `channels=1` since we are dealing with gray-scale images) and it gives back a vector of size `10` ($\hat{y}^{(i)} \in \Bbb{R}^{1 \times 10}$). We then pick the index with the highest value as the predicted target from the network.

### Feature Maps & Weight Sharing

> Classical work in visual pattern recognition has demonstrated the advantage of extracting local features and combining them to form higher-order features. Such knowledge can be easily built into the network by forcing the hidden units to combine only local sources of information. Distinctive features of an object can appear at various locations on the input image. Therefore it seems judicious to have a set of feature detectors that can detect a particular instance of a feature anywhere on the input place. Since the precise location of a feature is not relevant to the classification, we can afford to lose some position information in the process. Nevertheless, approximate position information must be preserved, to allow the next levels to detect higher-order, more complex features ([Fukushima 1980](https://www.cs.princeton.edu/courses/archive/spr08/cos598B/Readings/Fukushima1980.pdf), [Mozer 1987](https://psycnet.apa.org/record/1987-98557-004)).

We want our network to be comprised of many feature detectors. In the past, we used to do feature extraction by hand using manual extractors. In the case of images, low-level features consist of edges, colors, lines (...). We want to design a network capable of learning its extractors to detect these features then combine them and extract more abstract features like shapes, all to output the final vector indicating the digit.

The distinctive features of an image can appear at various locations. for example, in an image of a cat, we expect to detect two ears at various places, same with detecting leaves within a tree image, since we are more interested in the presence/absence of these features to classify the image category, their exact locations in the map are not relevant to classification. **This assumption will allow us to compress the size of the activation maps** without worrying much about location data that will be lost along the way.

> The detection of a particular feature at any location on the input can be easily done using the "weight sharing" technique. Weight sharing was described in [Rumelhart et al (1986)](http://www.cs.toronto.edu/~hinton/absps/naturebp.pdf) for the so-called T-C problem and consists of having several connections (links) controlled by a single parameter (weight). It can be interpreted as imposing equality constraints among the connection strengths. This technique can be implemented with very little computational overhead.

We will use the concept of **weight sharing** to detect low-level features at any location. Here, we will introduce two concepts:

1. **Kernel or Filter**: a square matrix of learnable weights, we use it to detect specific features in any part of the image.
2. **Activation**: The result of the filter sliding over the whole input.

Let's take a look at this example:

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/9UFISkH.png" style="border:0;">
  <figcaption style="font-size:12px; color:'#484848'">From <a target="_blank" href="http://www.cs.cmu.edu/~aharley/">Adam Harley</a>: <a href="https://scs.ryerson.ca/~aharley/vis/conv/flat.html" target="_blank">2D Visualization of CNNs</a>.</figcaption>
</figure>

In the figure, we have an input image with the number `1`. In the first layer, we have `6` activation maps that correspond to `6` kernels. Each kernel is responsible for highlighting a specific visual feature over the whole input image in its activation. The kernels are of size `5x5`. light squares represent big values, as we can see, the showcased filter detects diagonal sudden changes in input pixels. Other filters detect horizontal lines, vertical lines, corners, and so on.

In this example, the network has already been trained. As the input propagates to later layers, the size (width & height) of the activation maps get compressed in favor of increasing the number of activation maps to detect more abstract features until we get a single vector that exactly tells us which digit is present in the input image.

> Weight sharing not only greatly reduces the number of free parameters in the network but also can express information about the geometry and topology of the task. In our case, the first hidden layer is composed of several planes that we call _feature maps_. All units in a plane share the same set of weights, thereby detecting the same feature at different locations. Since the exact position of the feature is not important, the feature maps need not have as many units as the input.

The kernels, or filters, don't take in all image pixels at once, instead, each kernel outputs a feature value that corresponds to an input region and keeps sliding to cover the whole image to finally produce an activation map. as shown in the following example:

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/ouCeMH6.png" style="border:0;">
  <figcaption style="font-size:12px; color:'#484848'"><a target="_blank" href="http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/">Source: Understanding CNNs for NLP</a></figcaption>
</figure>

In this example, the kernel weights are $K=\begin{bmatrix}1 & 0 & 1 \newline 0 & 1 & 0 \newline 1 & 0 & 1 \end{bmatrix}$. If the input image is $X$, the output value at pixel $(m,n)$ is $Y_{m,n}$, the kernel is $K$, and the kernel size is $s$, then:

<div style="text-align:center">
$$Y_{m,n}=\sum_{i=0}^{s-1}\sum_{j=0}^{s-1}K_{i,j}X_{m+i,n+j}$$
</div>
    
By doing this, we're not using fully-connected neurons anymore but what is called **weight sharing**. Hence, weight sharing greatly reduces the number of free parameters, the first hidden layer is composed of several planes that we call **feature maps**.

### Network Architecture

<div style="text-align:center"><img style="width:66%;" src="https://i.imgur.com/3kCP00h.png" /></div>

> The network is represented in Figure 2 (_above_). Its architecture is a direct extension of the one proposed in [LeCun 1989](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.476.479&rep=rep1&type=pdf). The network has three hidden layers named `H1`, `H2`, and `H3`, respectively. Connections entering `H1` and `H2` are local and are heavily constrained.

The proposed network is comprised of three consecutive layers: `H1`, `H2`, and `H3`. We should note that `H3` doesn't have heavily constrained connections because it is a fully connected layer. The activations of `H1` and `H2` come from using the mentioned square kernels of weights but not `H3`.

> `H1` is composed of 12 groups of 64 units arranged as 12 independent `8x8` feature maps. These 12 feature maps will be designated by $H_{1,1},H_{1,2},\dots,H_{1,12}$. Each unit in a feature map takes input on a `5x5` neighborhood on the input plane. For units in layer H1 that are one unit apart, their receptive field (in the input layer) are two pixels apart. Thus, the input image is undersampled and some position information is eliminated. A similar two-to-one undersampling occurs going from layer H1 to `H2`. The Motivation is that high resolution may be needed to detect the presence of a feature, while its exact position need not be determined with equally high precision.

`H1` uses `12` kernels of size `5x5` that produce `12` activation maps noted: $H_{1,1},H_{1,2},\dots,H_{1,12}$. We know that the input is an image of size `16x16` but how come the activations are of size `8x8`? the reason is that instead of sliding the kernel over each possible position (moving by one pixel), we move by 2, this "sliding step" parameter is commonly referred to as the <b>stride</b>.

Using a stride of `2` and adding another `2` placeholder **padding** pixels at the margins of the image (will be mentioned later) produce the final activations of `8x8`, we essentially halved the size of the input image but we also get `12` feature maps that correspond to the `12` kernels. The following figure showcases the input image in green and one activation map in blue, we can see how the kernel is sliding over the image to fill its activation map:

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/SJMFV4d.png" style="border:0;width:66%;">
  <figcaption style="font-size:12px; color:'#484848'">Produced using <a target="_blank" href="https://github.com/vdumoulin/conv_arithmetic">conv arithmetic</a></figcaption>
</figure>

> It is also known that the kinds of features that are important at one place in the image are likely to be important in other places. Therefore, corresponding connections on each unit in a given feature map are constrained to have the same weights. In other words, each of the 64 units in $H_{1,1}$ uses the same set of 25 weights. Each unit performs the same operation on corresponding parts of the image. The function performed by a feature map can thus be interpreted as a nonlinear subsampled convolution with a 5x5 kernel.

Essentially, a kernel's weights are fixed while the kernel is sliding over the input image to calculate its activation values, this is essential to the task of detecting one visual feature all over the input image. A kernel can only detect one visual feature in a small region (`5x5`), we can think of its activation output as a heat map showcasing where that feature is present (re-visit the 2D Conv visualization figure).

> Of course, units in another map (say $H_{1,4}$) share another set of 25 weights. Units do not share their biases (thresholds). Each unit thus has 25 input lines plus a bias. Connections extending past the boundaries of the input plane take their input from a virtual background plane whose state is equal to a constant, predetermined background level, in our case, is `-1`. Thus, layer H1 comprises `768` units (`8x8x12`), `19,986` connections (`786x26`), but only `1068` free parameters (`768` biases plus `25x12` feature kernels) since many connections share the same weight.

An important thing to note, is that in `Pytorch`, each kernel has only `1` bias parameter and it is shared between the units of the activation map. According to this [answer](https://datascience.stackexchange.com/questions/11853/question-about-bias-in-convolutional-networks), a single bias per kernel makes sense because biases shouldn't depend on the location of the output value in the activation map, as we're using weight sharing to detect distinct features, each virtual neuron (filter) should have `1` bias. Also `PyTorch` uses a default padding value of `0` (`-1` in the paper).

All in all for `H1`, we have `12` kernels each of size `5x5` & having 1 bias parameter, for a total of `312` free parameters (`<<1068`), let's verify this:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
import torch.nn as nn
```

```Python
H1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=2)
```

```Python
[1]: sum(p.numel() for p in H1.parameters() if p.requires_grad)
312
```

> Layer `H2` is also composed of `12` features maps. Each feature map contains 16 units arranged in a `4x4` plane. As before, these feature maps will be designated as $H_{2,1},H_{2,2},\dots,H_{2,12}$. The connection scheme between `H1` and `H2` is quite similar to the one between the input and `H1`, but slightly more complicated because `H1` has multiple two-dimensional maps. Each unit in `H2` combines local information coming from `8` of the `12` different feature maps in `H1`. Its receptive field is composed of eight `5x5` neighborhoods centered around units that are at identical positions within each of the eight maps. Thus, a unit in `H2` has `200` inputs, `200` weights, and a bias.

For `H2`, we use another `12` kernels but the input to H2's kernels is different in size, `H1`'s kernels slided over one gray-scale image (1 channel). For `H2`, we use `8` maps from `H1`. So we use kernels of size `8x5x5`. In our context, to [understand](https://stats.stackexchange.com/questions/269893/2d-convolution-with-depth) how a kernel of size `8x5x5` convolve over an input of size `8x8x8`, we take every kernel plane and dot it with its corresponding input plane region to get 1 value, we do that for all 8 kernel planes to get a total of 8 values, we **sum them and add the bias**, ofcoures we do that for all regions to get `H2`.

> Once again, all units in a given map are constrained to have identical weight vectors. The eight maps in `H1` on which a map in `H2` takes its inputs are chosen according to a scheme that will not be described here. Connections falling off the boundaries are treated like as in `H1`. To summarize, layer `H2` contains `192` units (12 times 4 by 4) and there is a total of `38,592` connections between layers `H1` and `H2` (192 units times 201 input lines). All these connections are controlled by only `2592` free parameters (12 feature maps times 200 weights plus 192 biases).

Since how the `8` `H1` maps are picked will not be explained in the paper, we will use all `12` feature maps. To produce `H2`, we will use `12` `8x5x5` kernels for a total of `12x((12x5x5)+1)=3612` free parameters (`+1` is one bias per kernel). Let's verify this with `PyTorch`:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
H2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=2)
```

```Python
[2]: sum(p.numel() for p in H2.parameters() if p.requires_grad)
3612
```

> Layer `H3` has `30` units, and is fully connected to H2. The number of connections between `H2` and `H3` is thus 5790 (30 times 192 plus 30 biases). The output layer has `10` units and is also fully connected to `H3`, adding another `310` weights. In summary, the network has `1256` units, `64,660` connections, and `9760` independent parameters.

`H3` has `30` units (virtual neurons) and is fully connected to `H2`, hence we will flatten `H2` to directly connect it to `H3`'s parameters, we know that `H2` is comprised of `12` `4x4` planes, after flatenning, we get `12x4x4=192` input features, so H3 has `(192x30)+30=5790`, let's verify this in `PyTorch`:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
H3 = nn.Linear(in_features=192, out_features=30)
```

```Python
[3]: sum(p.numel() for p in H3.parameters() if p.requires_grad)
5790
```

# Experimental Environment

> All simulations were performed using the backpropagation simulator SN ([Bottou and LeCun 1988](https://leon.bottou.org/publications/pdf/sn-1988.pdf)) running on a [SUN-4/260](https://en.wikipedia.org/wiki/Sun-4).

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/PBGcRnm.png" style="border:0;">
  <figcaption style="font-size:12px; color:'#484848'"><a target="_blank" href="https://leon.bottou.org/projects/neuristique">SN</a>: Developed by Yann LeCun and Leon Bottou.</figcaption>
</figure>

> The nonlinear function used at each node was a scaled hyperbolic tangent. Symmetric functions of that kind are believed to yield faster convergence, although the learning can be extremely slow if some weights are too small ([LeCun 1987](https://www.researchgate.net/publication/216792893_Modeles_connexionnistes_de_l'apprentissage)). The target values for the output units were chosen within the quasilinear range of the sigmoid. This prevents the weights from growing indefinitely and prevents the output units from operating in the flat spot of the sigmoid. The output cost function was the mean squared error.

As an activation function, we will use the hyperbolic tangent, defined in the `PyTorch` [documentation](https://pytorch.org/docs/stable/nn.html?highlight=tanh#torch.nn.Tanh) as follows:

<div style="text-align:center;">
$$\sigma(x) = tanh(x) = \frac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$$
</div>

We can now update the formula for the convolution operation, by also considering the activation function $\theta$ and the kernel bias $b_{K}$:

<div style="text-align:center">
$$Y_{m,n}=\sigma(b_{K} + \sum_{i=0}^{s-1}\sum_{j=0}^{s-1}K_{i,j}X_{m+i,n+j})$$
</div>

Since target values weren't specificied, we will stick to **1**-hot encoding for the targets. An interesting decision is to pick mean squared error as a loss function. MSE calculates a geometric distance between a model prediction and the target value and is commonly used in regression tasks, for classification, we commonly use cross entropy. We will stick to MSE to compare the metrics. For a single image $X_i$, MSE can be expressed as:

<div style="text-align:center;">
$$J(\theta)=(y^{(i)}-\hat{y}(\theta,X^{(i)}))^2$$
</div>
    
> Before training, the weights were initialized with random values using a uniform distribution between $\frac{-2.4}{F_{i}}$ and $\frac{+2.4}{F_{i}}$, where $F_{i}$ is the number of inputs (fan-in) of the unit to which the connection belongs. This technique tends to keep the total inputs within the operating range of the sigmoid.

Since we'll be using `PyTorch`, we must note that it uses uniform [kaiming initialization](https://arxiv.org/pdf/1502.01852.pdf) for its layers.

> During each learning experiments, the patterns were repeatedly presented in a constant order. The weights were updated according to the so-called stochastic gradient or "on-line" procedure (updating after each presentation of a single pattern) as opposed to the "true" gradient procedure (averaging over the whole training set before updating the weights). From empirical studies (supported by theoretical arguments), the stochastic gradient was found to converge much faster than the true gradient, especially on large, redundant databases. It also finds solutions that are more robust.

In summary, we'll be feeding the images one-by-one (`batch size = 1`) and we'll use the stochastic gradient to update the parameters of the network.

> All experiments were done using a special version of Newton's algorithm that uses a positive, diagonal approximation of the Hessian matrix ([LeCun 1987](https://www.researchgate.net/publication/216792893_Modeles_connexionnistes_de_l'apprentissage); [Becker and LeCun 1988](http://yann.lecun.com/exdb/publis/pdf/becker-lecun-89.pdf)). This algorithm is not believed to bring a tremendous increase in learning speed but it converges reliably without requiring extensive adjustments of the parameters.

Although the proposed optimizer is not available in `PyTorch`, using second order approximators and hessian estimators to accelerate training convergence was very popular. Newton's second order method is derived by making a second order Taylor series approximation of $J(\theta)$ around $\theta_{k}$ (a quadratic approximation for example), then we can solve for the direction/acceleration of descent by first obtaining the gradients $g_k$, the Hessian $H_k$ and solve the linear system $H_{k}d_{k}=-g_{k}$ ([Lecture on optimization in DL](https://www.youtube.com/watch?v=0qUAb94CpOw)). For us, we'll be using normal `SGD`.

Here is the model:

```Python
import pytorch_lightning as pl
```

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
class DigitRec(nn.Module):
    def __init__(self):
        super(DigitRec, self).__init__()

        # Layers
        self.h1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.h2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.h3 = nn.Linear(in_features=192, out_features=30)
        self.output = nn.Linear(in_features=30, out_features=10)

        # Activation
        self.tanh = nn.Tanh()

    def forward(self, X):
        y = self.tanh(self.h1(X))
        y = self.tanh(self.h2(y))
        y = self.flatten(y)
        y = self.tanh(self.h3(y))
        y_hat = self.output(y)
        return y_hat
```

And finally, we integrate the model into a `lightning` trainer:

<div style="text-align:center; font-weight:bold;color:gray; font-size:13px">
    <a target="_blank" style="color:black;" href="https://github.com/Akramz/digit_rec">source</a>
</div>

```Python
class MNISTRecognizer(pl.LightningModule):

    def __init__(self):
        super(MNISTRecognizer, self).__init__()
        # Layers + Activation function
        self.h1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.h2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=2, padding=2)
        self.flatten = nn.Flatten()
        self.h3 = nn.Linear(in_features=192, out_features=30)
        self.output = nn.Linear(in_features=30, out_features=10)
        self.tanh = nn.Tanh()

    def forward(self, x):
        y = self.tanh(self.h1(x))
        y = self.tanh(self.h2(y))
        y = self.flatten(y)
        y = self.tanh(self.h3(y))
        y_hat = self.output(y)
        return y_hat

    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, F.one_hot(y.type(torch.LongTensor), 10).view(-1, 10).type(torch.FloatTensor))
        tb_logs = {
            'train_mse': loss,
            'train_log_mse': torch.log(loss),
            'error_rate': (y_hat.max(1)[1] != y.type(torch.LongTensor)).type(torch.FloatTensor)
        }
        return {'loss': loss, 'log': tb_logs}

    def validation_step(self, batch, batch_idx):
        """Not shown until epoch ends"""
        X, y = batch
        y_hat = self.forward(X)
        loss = F.mse_loss(y_hat, F.one_hot(y.type(torch.LongTensor), 10).view(-1, 10).type(torch.FloatTensor))
        logs = {
            'val_loss': loss,
            'val_log_mse': torch.log(loss),
            'val_error_rate': (y_hat.max(1)[1] != y.type(torch.LongTensor)).type(torch.FloatTensor)
        }
        return logs

    def validation_end(self, outputs):
        """Reporting done here"""
        avg_val_mse = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_log_mse = torch.stack([x['val_log_mse'] for x in outputs]).mean()
        avg_val_error_rate = torch.stack([x['val_error_rate'] for x in outputs]).mean()
        tb_logs = {'val_loss': avg_val_mse,
                   'val_log_mse': avg_val_log_mse,
                   'val_error_rate': avg_val_error_rate}
        return {'val_mse': avg_val_mse, 'log': tb_logs}

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=0.01)

    @pl.data_loader
    def train_dataloader(self):
        return get_digits(batch_size=1)[0]

    @pl.data_loader
    def val_dataloader(self):
        return get_digits(batch_size=1)[1]
```

# Results

<div style="text-align:center"><img style="width:66%;" src="https://i.imgur.com/DCFj44M.png" /></div>

> After each pass through the training set, the performance was measured both on the training set and testing set. The network was trained for `23` passes through the training set (`167,693` pattern presentations).

In our experiment, we split the dataset into two parts, training and validation, we set stochastic gradient descent as an optimizer with a learning rate of `0.01`, we measured training and validation MSE and error rate.

> After these `23` passes, the MSE averaged over the patterns and over the output units was $2.5 \times 10^{-3}$ on the training set and $1.8 \times 10^{-2}$ on the test set. The percentage of misclassified patterns was `0.14%` on the training set (10 mistakes) and `5%` on the test set (102 mistakes). As can be seen in figure 3 (above), the convergence is extremely quick and shows that backpropagation can be used on fairly large tasks with reasonable training times. This is due in part to the high redundancy of real data.

Here are our results:

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/xla0N2V.png" style="border:0;">
  <figcaption style="font-size:12px; color:'#484848'">Error rate and MSE on Train/Validation datasets.</figcaption>
</figure>

For `37` epochs, we get around `5%` error on the training set and `5%` error on the validation set, which is comparable to the paper's results. For loss convergence numbers, we get `0.01` loss on the training set and `0.01` for the test set, which is also comparable to the results of the paper. We shoud note that any generalization performance reporting should be either the validation or the final test set.

Although the two models and the source datasets are slightly different, we achieved similar results in generalization performance without reading the [early stopping](https://en.wikipedia.org/wiki/Early_stopping) point. We might explain the considerable difference in training metrics by us not having unit-level biases, reducing the total number of parameters by `~10%` (Another contributing factor is the choice of the optimizer and learning rate).

> In a realistic application, the user usually is interested in the number of rejections necessary to reach a given level of accuracy rather than in the raw error rate. We measured the percentage of test patterns that must be rejected in order to get a `1%` error rate on the remaining test patterns. Our main rejection criterion was that the difference between the activity levels of the two most active units should exceed a given threshold. The percentage rejection was then `12.1%` for `1%` classification error on the remaining (nonrejected) test patterns. It should be emphasized that the rejection thresholds were obtained using performance measures on the test set.

> Some kernels synthesized by the network can be interpreted as feature detectors remarkably similar to those found to exist in biological vision systems ([Hubel and Wiesel 1962](https://physoc.onlinelibrary.wiley.com/doi/pdf/10.1113/jphysiol.1962.sp006837)) and/or designed into previous artificial character recognizers, such as spatial derivative estimators or off-center/on-surround type feature detectors.

Through extensive research on cats, scientists discovered that certain nerve cells play the role of feature detection to detect vertical or horizontal lines for example. The interesting thing is that **our trained model optimized the kernel weight values to detect the same patterns**:

<div style="text-align:center;">
<span style="font-size:23px;">😿</span>
</div>
{{< youtube id="IOHayh06LJ4" >}}
<br/>

> Most misclassifications are due to erroneous segmentation of the image into individual characters. Segmentation is a very difficult problem, especially when the characters overlap extensively. Other mistakes are due to ambiguous patterns, low-resolution effects, or writing styles not present in the training set.

> Other networks with fewer feature maps were tried but produced worse results. Various fully connected, unconstrained networks were also tried, but generalization performances were quite bad. For example, a fully connected network with one hidden layer of `40` units (10,690 connections total) gave the following results: `1.6%` misclassification on the training set, `8.1%` misclassification on the test set, and `19.4%` rejections for `1%` error rate on the remaining test patterns. A full comparative study will be described in another paper.

### Comparison with Other Work

> The first several stages of processing in our previous system (described in [Denker et al 1989](http://papers.nips.cc/paper/107-neural-network-recognizer-for-hand-written-zip-code-digits.pdf)) involved convolutions in which the coefficients had been laboriously hand-designed. In the present system, the first two layers of the network are constrained to be convolutional, but the system automatically learns the coefficients that make up the kernels. Thus "constrained backpropagation" is the key to success of the present system: It not only builds in shift-invariance, but vastly reduces the entropy, the Vapnik-Chervonenkis dimensionality, and the number of free parameters, thereby proportionately reducing the amount of training data required to achieve a given level of generalization performance ([Denker et al 1987](https://pdfs.semanticscholar.org/33fd/c91c520b54e097f5e09fae1cfc94793fbfcf.pdf), [Baum and Haussler 1989](http://papers.nips.cc/paper/154-what-size-net-gives-valid-generalization.pdf)).

> The present system performs slightly better than the previous system. This is remarkable considering that much less specific information about the problem was built into the network. Furthermore, the new approach seems to have more potential for improvement by designing more specialized architectures with more connections and fewer free parameters (a network similar to the one described here with `100,000` connections and `2600` free parameters recently achieved `9%` rejection for `1%` error rate. that is about `30%` better than the best of the hand-coded-kernel networks).

Previously, researchers carefully specified the kernel weights before constructing the classification model, what's remarkable is that the only input knowledge we provide to the model is its constrained architecture (using weight sharing) to detect geometric features, the kernels are optimized via backpropagation and the results are slightly better than the previous hand-crafted systems.

> [Waibel (1989)](http://papers.nips.cc/paper/185-consonant-recognition-by-modular-construction-of-large-phonemic-time-delay-neural-networks.pdf) describes a large network (but still small compared to ours) with about `18,000` connections and `1800` free parameters, trained on a speech recognition task. Because training time was prohibitive (18 days on an Alliant mini-supercomputer), he suggested building the network from smaller, separately trained networks. We did not need such a modular construction procedure since our training times were "only" 3 days on a Sun workstation, and in any case, it is not clear how to partition our problem into separately trainable subproblems.

### DSP Implementation

<figure class="image" style="text-align:center;">
  <img src="https://i.imgur.com/RKWHnSh.png" style="border:0;width:66%">
  <figcaption style="font-size:12px; color:'#484848'">Pre-<a target="blank_" href="https://en.wikipedia.org/wiki/CUDA">CUDA</a>.</figcaption>
</figure>

> During the recognition process, almost all the computation time is spent performing multiply-accumulate operations, a task that digital signal processors (DSP) are specifically designed for. We used an off-the-shelf board that contains `256` kbytes of local memory and an AT&T DSP-32C general-purpose DSP with a peak performance of `12.5` million multiply-add operations per second on 32-bit floating-point numbers (25 FLOPS). The DSP operates as a coprocessor; the host is a personal computer (PC), which also contains a video acquisition board connected to a camera.

> The personal computer digitizes an image and binarizes it using an adaptive thresholding technique. The thresholded image is then scanned and each connected component (or segment) is isolated. components that are too small or too large are discarded; remaining components are sent to the DSP for normalization and recognition. The PC gives a variable-sized pixel map representation on a single digit to the DSP, which performs the normalization and the classification.

> The overall throughput of the digit recognizer including image acquisition is `10` to `12` classifications per second and is limited mainly by the normalization step. On normalized digits, the DSP performs more than `30` classifications per second.

# Conclusion

> We have successfully applied backpropagation learning to a large, real-world task. Our results appear to be at state of the art in digit recognition. Our network was trained on a low-level representation of data that had minimal preprocessing (as opposed to elaborate feature extraction). The network had many connections but relatively few free parameters. The network architecture and the constraints on the weights were designed to incorporate geometric knowledge about the task into the system because of the redundant nature of the data and because of the constraints imposed on the network, the learning time was relatively short considering the size of the training set. Scaling properties were far better than one would expect just from extrapolating results of backpropagation on smaller, artificial problems.

> The final network of connections and weights obtained by backpropagation learning was readily implementable on commercial digital signal processing hardware. Throughput rates, from the camera to classified image, of more than 10 digits per second were obtained.

> This work points out the necessity of having "network design" software tools that ease the design of complex, specialized network architectures.

# Acknowledgments

> We thank the U.S. Postal Service and its contractors for providing us with the database. The Neural Network Simulator SN is the result of a collaboration between Leon-Yves Bottou and Yann LeCun.
