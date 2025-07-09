
# Week 1: Intro to DL

### Terminology

**Deep Learning:** Act of training a neural network
**Neural Network:** A machine learning method, designed to make decisions in a fashion similar to the brain.
- Artificial: Any architecture of neurons layered together and interrelated. You have _input, hidden, output_ layers. 
- Convolutional: Useful for image data, perform convolutions on sections of the image to extract features of varying resolution. 
- Recurrent: Useful for sequence data by iterating on old data while also retaining memory of the past. 

### Notes

Consideration should be taken for the type of data your model is designed to work with (structured v unstructured), but the technique is obviously applicable to both. The evolution of deep learning has been driven greatly by scale, of:
- Data
- Computation: For brute-force accelerating the results from iterative experimentation
- Algorithms: E.g. switching activation functions, gradient descent. Therefore enabling faster computational speeds. 

# Week 2: Neural Network Basics

### Terminology

**Loss:** A means of measuring the error in prediction for a given training example.
**Cost:** The average loss across all training examples.
**Gradient Descent:** An algorithmic means of performing optimisation, by minimising a loss function. 
**Learning Rate:** $\alpha$, a hyperparameter we select in order to ensure stable convergence to the minimum of the cost surface. 
- If $\alpha$ is too high, then you will continually bounce around the minimum, and possibly even diverge from it if you go too far askew. 
- If $\alpha$ is too low, you will converge too slowly on the minimum, taking too much compute time. 

### Notes

We shall proceed motivated by the task of _binary classification_. This is an example of a task that could be learned by a neural network, and thus can be used to develop intuition about how it works. We can approach binary classification with _logistic regression_, which, like all of deep learning, is a supervised approach. 

The task may be framed as such. Given an object $X$ with true binary label $Y$ which can be described by a set of input features $x \in \mathbb{R}^n$, you wish to learn parameters $w \in \mathbb{R}^n, b \in \mathbb{R}$, such that your classifier will output $\hat{y}$ satisfying: 

$$\begin{align*} 
\sigma(z) = \frac{1}{1+e^{-z}} \; \; \text{[Sigmoid]}\\
\hat{y} P(Y=1|x) = \sigma(w^Tx + b) \approx P(Y = 1|x)\\
\end{align*}$$

We use the sigmoid function rather than a linear function to cap the values between 0 and 1, allowing us to determine a probability. 

We must next define a _loss function_ to measure the performance of our model on individual training examples. For this, we will use the _cross-entropy_ loss function (rather than standard loss functions like MSE or MAE). We seek a convex function so that we may find a single local minimum. We define the loss as:

$$\mathcal{L}(\hat{y}, y) = -(y\log\hat{y} + (1-y)\log(1-\hat{y}))$$

This is derived from taking the log of the following expression, which you can verify yourself by assuming that our model works and then comparing this to the true label:
$$P[Y=y \ | \ x] = \hat{y}^y(1-\hat{y})^{1-y}$$
We note that this is increasing, which is maintained under the logarithm, and then to turn it into a minimisation problem we negate the logarithm of this.  

Next we must perform _gradient descent_. The algorithm for this involves taking our known cost function, finding its gradient vector, and then moving in the opposite direction to it, scaled by the _learning rate_ to ensure good convergence. 

In this case, we know the function, so we shall provide the worked example.
We are going to redefine $x$ to now cover an entire training set with $m$ examples, making it instead into a matrix of dimension $n \times m$, for notational clarity. 

$$\begin{align*}
\mathcal{C} = \frac{1}{m}\sum_i^m \mathcal{L}(\hat{y}_i, y_i)
\end{align*}$$
With:
- $\mathcal{C}$ - The cost function
- $\hat{y}_i$ - The predicted label of example $x_i$
- $y_i$ - The true label of example $x_i$ 

We seek:
$$
\nabla \mathcal{C} = 

\begin{pmatrix}
\frac{\partial \mathcal{C}}{\partial w_1} \\ 
\vdots\\ 
\frac{\partial \mathcal{C}}{\partial w_n} \\
\frac{\partial \mathcal{C}}{\partial b}
\end{pmatrix}
$$
We must evidently use the chain rule. We intermediately define $z_i = w^Tx_i + b$. 
Intermediate steps such as the derivative of the sigmoid function are left as an exercise to the reader, which should be trivial given familiarity with the chain rule. 

$$
\begin{align*}
\frac{\partial \mathcal{L}}{\partial \hat{y}_i} &= -[\frac{y_i}{\hat{y}_i} - \frac{1-y_i}{1-\hat{y}_i}] \\

\frac{\partial \hat{y}_i}{\partial w_k} &= \frac{d\sigma}{dz_i} \cdot \frac{dz_i}{dw_k} \\
&= \sigma(1-\sigma) \cdot x_{ik}\\

\frac{\partial \hat{y}_i}{\partial b} &= \frac{d\sigma}{dz_i} \cdot \frac{dz_i}{db} \\
&= \sigma(1-\sigma)

\end{align*}
$$

Reminder: $\hat{y}_i = \sigma(z_i)$, for the middle partial derivative. 

We have now found the derivatives to chain together, and so we shall find the final necessary derivatives by multiplying them together in the appropriate fashion, an exercise which is useful for implementation.

Given $\nabla \mathcal{C}$, we know from multivariate calculus that this points in the direction of $\mathcal{L}$ increasing, and so to guarantee that we traverse antiparallel with the direction of greatest ascent, we negate the signs, and scale by the _learning rate_. In (vectorised) code, this will simply look like:

```Python
params -= lr * grad_C
```


This is process is trivial for known functions, and in the case of regression we are trying to fit a known function. However, considering a neural network as a set of:
- Weights $w^{(i)}$ per layer
- Biases $b^{(i)}$ per layer 
- Activation functions $a$ per layer. 

The principle is the same, but we need to find a way to automate the process for any architecture. The solution is to store a _computation graph_, where nodes represent expressions, and directed edges represent a root expression being the composition of two expressions. You can then in-bake derivatives for the known expressions your system will support; this is particularly pertinent for the activation functions, which introduce non-linearity in your system and allow for greater complexity to be picked up. 

By traversing the graph's edges in alternate directions, you can perform:
- _Forward passes:_ Actually computing chained expressions to produce the outputs of your neural network.
- _Backward passes:_ Computing the derivatives of expressions by the chain rule by following the graph back to its source nodes. 

# Week 3: Shallow NNs

### Notes

#### Architecture

Neural networks are comprised of three types of layer:
- **Input Layer:** One layer where the features/input data are passed to the network
- **Hidden Layer:** There may be as many of these as the architect deems necessary. These perform the main bulk of 'processing' in the neural network, taking in the results of the preceding layer, applying _weights, biases_ and an _activation function_ to the inputs to each neuron. 
- **Output Layer:** Where the results of the neural network are provided. These also have weights, biases and activations. 

Note: When counting layers we (reasonably) don't count the input layer. Or we can 0-index our counting to include it, they're equivalent. 

For each layer $i$ consisting of $L_i$ neurons, then for each neuron, $n$, the computation goes:
- $z_i = \sigma(w_i^Tz_{i-1} + b)$
- Here we assume that $\sigma$ is an activation function applied element-wise
- We also assume that:
	- $w_i$ is a weights matrix (alternatively viewed as a layer of a tensor) where $w_{ijk}$ is the weight of the  $j$-th input in $z_{i-1}$ that contributes to the $k$-th item in $z_i$.  
	- $b$ is a biases vector which adds a linear bias to the contributions of each neuron to the activation. 

The notational fluff starts to complicate the idea, but provides a lens into vectorising the operations by converting it to matrix and vector operations (which, of course, can be parallelised) by considering these linear transformations, pre-activation, as matrix operations. 

You can also _parallelise across multiple examples_, allowing you to drastically reduce training times, by compacting your input data into a matrix rather than multiple distinct vectors, and then using a similar intuition as one might when performing broadcasting for similar dimension arrays in Numpy (at least for the biases, since that would take less storage than homogeneous coordinates). 

Neural networks were originally analogised to the brain, where neurons only transmit a signal if they are provided with sufficient input to reach a certain threshold. We mimic this effect using an _activation function_. The more fitting idea is the simple fact that most real-world data is non-linear, so you need to introduce non-linear functions in order to capture input-to-output mappings that have sufficient expressivity. 

Some considerations about activations include:
- $\tanh$ is better than sigmoid due to the effect of centring the values on 0, except for binary classification at the output layer.  
- Different layers may have different activation functions!
- Both $\tanh$ and sigmoid fall short since you lose sensitivity for large values (vanishing gradient problem), which is where one may choose to use ReLU (which is normally quite optimal). 

For more on activation functions:
- [Datacamp Activation Functions](https://www.datacamp.com/tutorial/introduction-to-activation-functions-in-neural-networks)
- [v7 Blog Activation Functions](https://www.v7labs.com/blog/neural-networks-activation-functions)


_Notes on initialisation:_
- Weights in a neural network need to be initialised to non-zero values since otherwise you will be locked into 0 gradients for each back propagation step. 

#### Training

I will approach this from a slightly more mathematical lens since this is what I personally was seeking from the course. 

We've analysed forward propagation and its vectorization by reducing it to matrix operations in the previous section. To summarise, the recursive formulae are here:

$$\begin{align*}
a^{[1]} = \sigma^{[1]}(w^{[1]}X + b^{[1]}) \\
a^{[n]} = \sigma^{[n]}(w^{[n-1]}X + b^{[n]}) \\
\end{align*}$$

You may then terminate this at any point, depending on the chosen depth of your neural network. 

For back-propagation, we need to once again perform _lots_ of chain rule, but this too can be vectorized by making a computation graph that works on tensor objects, and by considering the Jacobian matrix of each layer, since you can treat each layer as a vector-valued function, and then multiply the Jacobians to implement the chain rule. Unfortunately I feel compelled to actually explain this to the poor soul who's reading this as a useful resource, so here you go :D. 


# Week 4:  Deep NNs

### Terminology

**Parameters:** Values that the model needs to learn in order to make predictions
**Hyperparameters:** Values that we control in order to optimise how effectively the model learns. They might include:
- Learning Rate, $\alpha$ 
- Optimiser algorithm
- Number of training iterations
- Number of hidden layers, $L$
- Number of hidden units, $n^{[l]}$
- Activation functions at each layer

### Notes

As one may intuitively expect, depth allows us to capture more expressivity and complexity in the functions that can be learned by our model, by introducing a deeper hierarchy of information that can be learned/composed between layers
- Convolutional Neural Networks: Edges -> Features(eyes, mouth, etc) -> Faces
- Audio: Low level audio waveforms -> Phonemes -> Words -> Sentences/Phrases

You may also think about it in terms of circuit theory. The informal result is that a small (low number of neurons per layer) $L$-layer deep net can compute functions that would take shallower networks exponentially more hidden units to compute.

Otherwise, the function of the network's forward and backward propagation is identical to how we described it before, in terms of the recursive definition of forward propagation, and in terms of how back-propagation requires you to recurse through the computation graph/tree of the neural network using the chain rule. 

Applied deep learning is empirical. This means that you will need to go through the process of tweaking your hyperparameters in order to find the point that minimises your cost, $\mathcal{C}$. There are slightly more systematic ways of doing this, but for now you just have to try it. 