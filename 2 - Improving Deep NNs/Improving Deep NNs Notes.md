
# Week 1 - Practical Aspects of Deep Learning

### Experimental Methodology

Machine Learning is an empirical science. While there are standard algorithms and workflows, it is ultimately on you to mess around and experiment in order to collect data and come to a conclusion on what the best reasonable model to solve your task is. 

We've seen this highlighted particularly regarding the notion of _hyperparameter_ selection. You can't guess them all right on the first go, and you'll need to do some exploring to work out what works best!

You have multiple datasets used across the span of a machine learning project:
- **Training Set:** You use this to train the parameters of each model you generate
- **Development Set:** You test each individual model's performance on the validation set to find out which one performs best, and is unseen. This is used for tuning hyperparameters. 
- **Test Set:** When you have a final model to evaluate, you measure its performance on this fully unseen bit of data. This allows you to have an unbiased estimate of how well the model generalises. 

You need to make the appropriate decision to split up your data in a fashion that facilitates sufficient testing ability. 

**NB:** Make sure that your dev and test sets come from the same distribution, since otherwise you might think that you're generalising well to your task domain from the dev set, and then you bring it to testing and you perform awfully. 

Note: The following is my own supplementation of the course content by shedding a more formal lens on some of the terminology used. 

Let $M$ be an _estimator_ for some parameter of an unknown (random) variable $X$, i.e. a function/variable which takes samples of $X$ and uses these to infer a parameter, $\theta$, determining $X$. We say that an estimator has:
 $$\text{Bias} = E[M - \theta] = E[M] - \theta$$
It is obvious that an _unbiased estimator_ is one that has $E[M] = \theta$, and it is also hopefully clear that bias is a measure of accuracy/systematic error. High bias is symptomatic of underfitting the estimator $M$ to your data. We may clearly treat models as estimators, given training data. 

**Variance** measures the difference in the predictions a model makes under different training sets. A high variance indicates that there is a sensitivity to noise/small details in the training data, and is symptomatic of overfitting to the training data, leading to poor generalisation. 

You can use the _train/dev set errors_ in order to diagnose the relative bias/variance of your model, like so:
- A _high training set error $\implies$ high bias_. I.e. the model hasn't learned how to do the task even on the training set)
- A _high dev set error_ $\implies$ _high variance_. I.e. the model hasn't learned the general principle behind the task, only memorised structure/patterns in the training data.

Such measurements are made on the predicate of first estimating the _optimal Bayes Error_, i.e. the level of human error in making predictions, and then using this as a benchmark for the level of error your model actually has. Typically you assume this is quite small. 

Given the intuition behind these errors, it then informs how to fix them:
- **Bias Problem:** Seeks to solve the fact that the model simply doesn't have enough complexity/training to capture the mapping between inputs and outputs. There is a _systematic error_. 
	- Bigger/Deeper network
	- Train it for longer
	- Better optimiser algorithms
	- Perhaps in an extreme case, try a different network architecture, since there are many domain specific ones that exist. 
- **Variance Problem:** Seeks to solve the fact that your model is very erratic and prone to lots of change in the training data. I.e. you need to minimise _random error_. 
	- Get more training data + performing Regularisation
	- Also consider trying a different architecture. 

You want to work on minimising bias before you fix variance. The idea is that you're trying to optimise the [Bias-Variance Tradeoff](https://www.ibm.com/think/topics/bias-variance-tradeoff). However, in the context of deep learning, you can usually always optimise both simultaneously by getting more data and using a bigger network. 

### Regularisation

**Regularisation:** A set of techniques to reduce the variance of a model, acting to prevent overfitting by opting for simpler representations/functions. 
#### Penalties

Given our original cost function, depending on parameters $\theta$:
$$\mathcal{C}(\theta) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}_i, y_i)$$
One way of performing regularisation by augmenting it with an additional _penalty term_, $\Omega(\theta)$ scaled by a new nonnegative hyperparameter $\lambda$, which determines the strength of the regularisation/penalty, giving a new cost:

$$\mathcal{C}'(\theta) = \frac{1}{m} \sum_{i=1}^m \mathcal{L}(\hat{y}_i, y_i) + \lambda \cdot \Omega(\theta)$$

| Name              | $\Omega(\theta)$                          |
| ----------------- | ----------------------------------------- |
| L2/"Weight decay" | $\frac{1}{2m}\sum_{l=1}^L\|w^{[l]}\|^2_F$ |
| L1                | $\frac{1}{m} \sum_{l=1}^L \|w^{[l]}\|$    |

The matrix norm given for L2 regularisation is known as the _Frobenius norm_, which is the sum of each element in the matrix squared. You can then calculate the change using matrix calculus to get $dw^{[l]} = B + \frac{\lambda}{m}w^{[l]}$, where $B$ is the same change calculated in regular back propagation. Due to the additional term, this is also called weight decay. 

You might seek to gather some intuition for this penalty term by looking to the similarities with Lagrange multipliers. What adding this term does is imposes a geometric constraint on the norm of the parameters being optimised to try and pull it as close as possible to the origin, thereby minimising complexity by keeping as many of the weights 0, or as close to it as possible. The L2 regulariser shines in particular since it enables the important weights to stay high. 

Penalty-based regularisation prevents overfitting by simplifying the internal mapping learned to be less non-linear, therefore reducing the ability to capture very curved or nuanced decision boundaries, which would be symptomatic of overfitting. 

#### Dropout 

**Dropout Regularisation:** Where you randomly deactivate nodes at each layer, re-selecting each set of hidden units on every training iteration. There are many ways to implement dropout:
- _Inverted Dropout:_ Deactivate certain nodes, and then divide the remaining live activations by the probability that they were doing to be kept, thus rescaling the contributions of the remaining nodes. 
- 

The intuition behind dropout is that each neuron can now no longer rely on a particular feature from the previous layer, since any one of them could be dropped out. Therefore, to learn optimally, the network has to spread out the weights, and so we invoke a similar effect to L2 regularisation. 

When considering the `keep_prob` hyperparameter, you will want to make it smaller for layers with larger weight matrices, which could be more prone to overfitting, and therefore invoking a stronger preventative dropout. 

A disadvantage of dropout is that $\mathcal{C}$ is no longer well defined, so you might not have it be monotonically decreasing. Experimentally you will simply want to compare the performances between having dropout on and off. 

#### More Methods

- **Data Augmentation:** This helps you get more data by taking your existing data and applying transformations of varying strength that would still preserve the prediction made. By training on this it allows your model to generalise better by learning from a more general set of examples. 
- **Early Stopping:** Stop training the neural network about halfway through the training iterations, or at least until your dev set error starts to increase again. This means that your model learns general patterns in the input data, not patterns specific to the training data. However, this couples the tasks of optimising the cost and preventing overfitting, meaning that it's difficult to orthogonalise your search of hyperparameter space. 

For more on regularisation:
- [IBM Regularisation Blog Post](https://www.ibm.com/think/topics/regularization)
- [Pinecone Regularisation Blog Post](https://www.pinecone.io/learn/regularization-in-neural-networks/#L1-and-L2-Regularization)
- Deep Learning, Goodfellow, Bengio, Courville. 

### Setting up Optimisation

One consideration you might want to make before training the network is _normalising data_, which will allow you to speed up the training of your neural network. You should ensure that you normalise the training and test data with $\mu, \sigma$ from the whole distribution rather the individual sample. 

Normalising your data will make the parameters you need to learn more spherical/symmetric, and therefore it will be easier to find minima via gradient descent, with a slightly larger learning rate, whereas for data with different scales, you need a smaller learning rate to account for the highest resolution data, increasing training times. 

Another problem to consider is that of _vanishing/exploding gradients_. Suppose your neural network is comprised of $L$ layers, with all biases, $b^{[l]} = \mathbf{0}$, weights $W^{[l]} = (1+\epsilon)\mathbf{I}$, for some $0 < |\epsilon|  \ll 1$, and activations as the identity function. Consider performing a forward pass through the network for some input vector $x$, up to the $n$th layer. Trivially, one would find that:
$$a^{[n]} = (1+\epsilon)^n \ x$$
We thus run into these problems:
- If $\epsilon$ is positive, then you get exponential growth in the resulting gradients in the number of layers $n$. I.e., the gradients _explode_. 
- If $\epsilon$ is negative, then you get exponential decay in the resulting gradients in the number of layers $n$. I.e., the gradients _vanish_. 

These problems will only become more exacerbated in practice with non-zero biases, and larger weights, and so you must mitigate by carefully initialising the weights. You can do this by fixing the mean ($\mu = 0$) and variance of the distribution from which the weights are sampled. The following activations work with the following variance initialisations:
- ReLu: $Var[W] = \frac{2}{n^{[l-1]}}$
- $\tanh : Var[W] = \frac{1}{n^{[l-1]}}$. Known as Xavier initialisation. 
- Bengio Initialisation: $Var[W] = \sqrt{\frac{2}{n^{[l-1]} + n^{[l]}}}$

You want to perform _gradient checking_ to verify your backpropagation implementation. You should be sure to take a two-sided difference (of magnitude $\varepsilon$) in order to reduce your error margin from $O(\varepsilon)$ to $O(\varepsilon^2)$. You should inspect on a component-by-component basis. You should only do this when debugging, remembering to include regularisation terms. Note that this doesn't work with dropout due to the random dependencies. Your implementation might also be correct for a specific locus of parameter values, so you should take care to verify it with random initialisations. 

# Week 2 - Optimisation Algorithms

### Gradient Descent and Batches

In traditional gradient descent, we've already shown you can efficiently compute across $m$ examples, to capture your entire training set. But if your training set is absurdly large (to the point that even vectorization doesn't fully stem the compute time), you might want to try to speed up the gradient descent.

**Mini-Batch Gradient Descent:** Take small subsets of your training set and allow gradient descent to occur from training on subsets, rather than waiting for your entire training set to be processed. 

```python

parameters = initialise_parameters()
epochs = ...
n_batches = ...
batches = split_data(train_set, n_batches)

for epoch in range(epochs):
	for batch in batches:
		cache = forward_propagation(data[b])
		cost = compute_cost(cache)
		grads = backward_propagation(cache)
		parameters -= lr*grads
```

The _epochs_ and _number of mini batches_ are now new hyperparameters for you to tune. 

You'll observe in training, that there will now be a downwards trend with significantly more noise to it, since you're effectively exposing it to new data which the model might not yet be proficient in handling. You will want to be wise in choosing your mini-batch size however, so that you can maximise the speed up. 

- **Batch Gradient Descent:** When the mini-batch size is equal to the size of the entire training set. This takes too long per iteration, given a very large training set. 
- **Stochastic Gradient Descent:** When the mini-batch size is 1, and every example is its own batch. This has the most noise to it, and will never converge due to that effect. You can mitigate the noise with a smaller learning rate, but you lose all the speedup from vectorisation. 

These two extremes are clearly unsuitable, so you want something in the middle. If your training set is small then you can opt for batch descent, but otherwise, it is advisable to pick a batch size which is a power of 2, which fits in GPU memory. 

### Exponentially Weighted Moving Averages 

This might be familiar if you've looked at implementations of the _STRF_ scheduling algorithm. The general premise is to take measurements at discrete time steps and combine the old average with the new incoming data in the following fashion:

Let $V_t$ be the average at any time increment $t$, and $\theta_t$ be the incoming data at timestep $t$. Then we define the exponentially weighted average as:

$$V_t = \beta V_{t-1} + (1-\beta)\theta_t$$
You select $\beta$ depending on how you judge the relative importance of the old data against the new data. You can interpret $V_t$ as approximately averaging over the last $\frac{1}{1-\beta}$ measurements. 

Let's dwell on this to establish some properties. We can expand the recursive definition of the formula (initialised with some value $V_0$) to yield the following:

$$V_t = \beta^{t+1}V_0 + \sum_{i=0}^{t-1} \beta^i \ (1-\beta) \ \theta_{t-i}$$

We can consider each piece of new information from timestep $t-k$ as having weight $w_k$. Evidently from the above formula, we observe that $w_k = \beta^k (1-\beta)$. We can find the average lag, $L$, for a piece of information contributing to the mean by the following formula:

$$\begin{align*}
\mathbb{E}[L] &= \sum_{k=0}^{\infty} k\cdot w_k \\
&= \sum_{k=0}^{\infty} k\cdot (1-\beta)\beta^k \\
&= \frac{\beta}{1-\beta}
\end{align*}$$

You can consider this mean in relation to the variable $\text{Geo}(1-\beta) - 1$, allowing you to intuit this as _a system which forgets with probability $1 - \beta$_. The last remembered data point would be $x_{t-K}, K \sim \text{Geo}(1-\beta) - 1$. 

Note: At this point, I confess that I'm trying hard not to overthink the off-by-one in the geometric variable, and I would recommend to a less qualified reader to do the same, this off-by-one interpretation maintains the notion of the geometric distribution as the number of trials until success, rather than counting the number of failures until success, though both are fair interpretations. For me, I've personally always been taught the former and thus wish to retain that. 

Now to find the _effective window_ this covers, we shall _consider the information as undergoing exponential decay in weight_. We shall say that a piece of information is significant if its current weight, $w_k$ remains above $\frac{1-\beta}{e}$. 

$$\begin{align*} 
(1-\beta) \beta^k &= \frac{1}{e} (1-\beta) \\
\beta^k &= \frac{1}{e} \\
k \ln \beta &= -1 \\
k &= \frac{-1}{\ln \beta} \\
k &\approx \frac{1}{1-\beta} \; \; [\text{By Taylor Expansion about } x = 1 ] \end{align*}$$
Thus, the lag $k$ within which a piece of information remains significant is as given above, giving our proposed 'viewing window'. 

When using exponential averaging, you run into an error for _early iterations_, whereby if the value of $1- \beta$ is small, then you will _introduce a systematic underestimate of early values_. To rectify this, you could choose to output $\frac{V_t}{1-\beta^t}$ on each iteration, which will turn it into a standard average of the incoming $\theta_i$s. 

**Note:** $V_t$ still obeys the proposed formula, you're simply applying a correction algorithm before returning the average. 

### Momentum

**Momentum:** Take an exponentially weighted average of your gradients, and use this to update your parameters instead. 

Regular gradient descent slows you down since you run into the issue of your path being vaguely oscillating, preventing you from using a high learning rate since you run the risk of diverging on a given oscillation.  

Momentum will help you because by averaging out directions, it dampens any oscillations in the path, meaning that you converge along a more direct path to your minimum. The 'random walk'-like path you would've taken now has far less variance. 

A nice analogy (for those with a mind for physics) is to imagine a ball rolling down a convex surface. If we consider the term:

$$V_{dW,t} = \beta V_{dW, t-1} + (1-\beta)dW$$
We can consider this term as being the resultant acceleration on the ball. The first half is the current velocity, limited by some amount of friction, and the second term is the acceleration due to gravity at that point on the surface. 

$\beta = 0.9$ is a common choice for the hyperparameter, but you might need to re-tune $\alpha$ in response to any changes of $\beta$.

### RMSProp

**RMSProp:** This is an alternative optimisation technique to momentum. It achieves this with the following set of equations to maintain a moving average of the squared gradients. 

$$\begin{align*}

&s_{dW,t} = \beta s_{dW,t-1} + (1-\beta)dW^2 \\

&w \leftarrow w - \alpha \frac{dW}{\sqrt{s_{dW,t} + \epsilon}} \\

\end{align*}$$
**NB:** The equations given above are only adjusting the weight, but the biases must also be updated too. 

This once again has a similar effect to dampening the oscillations as last time:
- If the gradients are large, such as due to oscillations,  then the moving average will be large, meaning that you divide the calculated gradient by a large number, dampening these oscillations. 
- If gradients are small, then then you divide the gradient by a very small moving average drastically accelerating the learning. 

You need the little $\epsilon$ term to prevent a potential divide-by-zero error. 

### Adam 

**Adam:** Adaptive Moment Estimation. Takes the best of both worlds from momentum (the first moment) and RMSProp (the second moment) and combines them to achieve a well-generalising optimisation algorithm. 

We now calculate $V_{d\theta,t}$ and $s_{d\theta,t}$,  _under bias correction_. We then update under the parameters $\theta$ under the formula:

$$ \theta \leftarrow \theta - \alpha \frac{V_{d\theta}} { \sqrt{s_{d\theta} + \epsilon}}$$
We note the hyperparameters we have here:
- $\alpha$, learning rate. This **definitely** needs tuning. 
- $\beta_1 (= 0.9)$. Used to calculate $V_{d\theta,t}$ 
- $\beta_2 (=0.999)$. Used to calculate $s_{d\theta ,t}$. 
- $\epsilon = 10^{-8}$. 

You could tune the $\beta_i$s, but it's not really that important. You also don't necessarily need to tune $\epsilon$. 

### Learning Rate Decay

This is motivated by the desire to ensure better convergence on the exact minimum, rather than bouncing around the minimum while never reaching it. As you get closer to the minimum (your gradients get smaller, or more training iterations pass), you increase the resolution with which you traverse the loss surface, allowing you to more finely reach the minimum. 

There are a number of ways of implementing this, such as:

- _Inverse decay_ $$\alpha = \frac{1}{1 + \lambda n_e}\alpha_0$$
- _Exponential Decay_ $$\alpha = \lambda^{n_e}\alpha_0$$
- _Inverse Root_
- _Discrete Staircase_
- _Manual Decay_. This is where you, the scientist, manually interject and reset the learning rate as you please. The use context is when you're training for hours or even days, across a small number of models. 

In all cases, we have:
	- $\lambda$: The decay rate you choose
	- $n_e$: The epoch number you're currently on
	- $\alpha_0$: The initial learning rate.

# Week 3 - Hyperparameter Tuning + Batch Normalisation

This week technically also explores TensorFlow but I don't personally see the pertinence to include code snippets or notes when you'll learn how to use it in the programming assignments, and can have access to the docs. 
### Hyperparameter Tuning

There is a hierarchy of importance regarding which hyperparameters need tuning, though there are many varieties of intuition on which ones are important to explore. Andrew suggests:
- Learning Rate
- $\beta$ in momentum, Hidden Units, Mini-Batch Size
- Layers and Learning Rate Decay (method + decay rate)

In early machine learning, _systematic grid sampling_ used to be used, which certainly makes sense, but is highly inefficient, particularly when you have a high number of hyperparameters. Instead, pick a certain number of samples, and _randomly sample the parameters on each go_ . With the grid method, you'll actually have sampled less values of each hyperparameter with the same number of trials, which is clearly a less rich exploration of the hyperparameter space. 

You should also consider a _coarse-to-fine_ search methodology, increasing the resolution of your search as you gain more insight to where the optimal set of hyperparameters might be. 

There is also a problem of _picking the appropriate scale_ to search for hyperparameters on, since you don't want to waste resources unevenly searching the range of values:
- For $n^{[l]}$ a uniform linear scale would be appropriate
- For $\alpha$ or $1-\beta$, which can vary on the order of magnitude, and are very sensitive hyperparameters, you would want to sample along a logarithmic scale and then pick that. 

You should be considerate of the fact that due to changes in your workflow, your hyperparameters could go stale. It's good practice to re-test them every few months or so. Pick an appropriate scale :). 

There are two big schools of thought on model-training, which are restricted by your available computational resources:
- Pandas: Babysit one model at a time, searching hyperparameter space sequentially until you reach a good one.
- Caviar: Train loadssss of models in parallel. And then see which regions of hyperparameters do the best! 

### Batch Normalisation

**Batch Normalisation:** A technique to improve the stability of a neural network, and to accelerate training by making the hyperparameter search easier. 

Suppose we're inspecting some layer $l$ in a deep neural network. Consider the linear values (pre-activation) at that layer. There is some debate about whether we should do this pre or post activation, but we shall adopt the pre-activation school of thought for now. We shall take:

$$\begin{align*}
&\mu = \frac{1}{m}\sum_i z^{(i)} \; \;   \; \; 
&\sigma^2 = \frac{1}{m}\sum_i (z^{(i)} - \mu)^2 \\
\end{align*}$$

$$\begin{align*} 
z_{\text{norm}}^{(i)} &= \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
\\
\tilde{z}_{\text{norm}}^{(i)} &= \gamma z_{\text{norm}}^{(i)} + \beta
\end{align*}$$

The reason we perform the affine transformation is because with certain activation functions, centring the data strictly around 0 and having too small a variance, might not allow us to maximally make use of the non-linearity, so we can increase the variance or re-centre the data as desired. **NB:** $\beta,\gamma$ are learnable vector parameters for the model.

When using batch-norm on a layer, we can eliminate the bias parameter for that layer, since the normalised version will be re-centred on 0, and then you can use $\beta^{[l]}$ to re-calibrate the mean. 

**Covariate Shift:** The problem of a data distribution changing from training to testing/application, while the learned mapping/function stays the same, meaning that your function is not necessarily suitable to predict the new distribution. 

A layer in the middle of the network receiving input from the previous layer effectively only sees the features it receives. Everything else behind it might as well be a black-box, relative to it. What happens during training, however, is that as the parameters of the black-box get optimised, there is a covariate shift in the received inputs from the previous layer. Batch normalisation reduces the coupling between the black-box and the parameters the third layer learns, since now the received data will always be constrained (slightly) by its mean and variance, thereby reducing the amount of covariate shift which can occur, by having some stability in the distribution of the output features from the previous layer. 

Batch-normalisation _also acts as a slight regularisation method_, similar to dropout, by introducing some noise by applying the affine transformation to each hidden layer's activations. You might want to use dropout in cohesion with this. As a slight aside, regularisation effects of dropout are reduced from larger mini-batch sizes. 

At test-time, you might not have mini-batches to process in parallel. You need a different way to conceive $\mu, \sigma^2$ at each layer. You normally do this by estimating them with an exponentially weighted average across mini-batches. At test-time, you use these averages to fill in for the mean and variance. 

### Multi-Class Classifiers

Here, you want to generalise the notion of _logistic regression_ to _softmax regression_. The distinction here, hopefully trivially, is that the output layer will no no longer be one value, but instead be a $C$-vector, where $C$ is the number of classes we're trying to distinguish between. 

Here, the activation at the output layer will be the softmax function, which is given by:

$$\text{softmax}(z) = \frac{e^{z^{[l]}}}{\sum_{i=1}^C e^{z_i^{[l]}}}$$

In other words, you normalise the output vector such that all entries sum to 1, meaning that each entry now represents the probability that a set of input features corresponds to that class. This enables you to learn more complex non-linear decision boundaries for multiple classes.

Softmax is named in contrast to the "hardmax" function which places a 1 in the place of the maximum entry. It generalises the sigmoid activation function to more than 2 classes. If we briefly review the sigmoid function:

$$\sigma(z) = \frac{1}{1+e^{-z}} = \frac{e^z}{e^z+1}$$
<Insert inspection/intuition here/> 

The loss function we want to use to train a softmax classifier is:

$$\mathcal{L}(\hat{y},y) = -\sum_{c=1}^C y_c \log \hat{y_c}$$
This makes sense, since to minimise the expression, the only thing that can be changed is $\hat{y}_\hat{c}$, where $\hat{c}$ is the correct identification. This turns out to be a form of maximum likelihood estimation, similar to logistic regression. We are selecting the parameters of our neural network to maximise the likelihood that we identify the input features as corresponding to class $\hat{c}$. 

$\text{RTP: } \frac{\partial J}{\partial z^{[l]}} = \hat{y} - y$

First we clarify our definitions:
$$\begin{align*}

\hat{y} &= \text{softmax}(z) \\ 
J &= \frac{1}{m} \sum_i \sum_j y_j \log \hat{y}_j \\

\end{align*}
$$
We note that we're taking an element-wise derivative in the definition. 

$$\begin{align*}

\frac{\partial J}{\partial z^{[l]}} &=  \frac{\partial J}{\partial \hat{y}} \cdot  \frac{\partial \hat{y}}{\partial z^{[l]}} \\

\frac{\partial \hat{y}}{\partial z^{[l]}} &= \hat{y} - \hat{y}^2


\end{align*}$$

It's nice to be able to prove this result, however, it is certainly not necessary, since deep learning frameworks can do the backpropagation and auto-differentiation for you! For example, TensorFlow requires us only to specify the forward propagation step. 