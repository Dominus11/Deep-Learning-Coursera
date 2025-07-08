
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

### Terminology

### Notes



# Week 4:  Deep NNs
