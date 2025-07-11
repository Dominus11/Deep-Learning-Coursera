
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

# Week 3 - Hyperparameter Tuning + Batch Normalisation

This week technically also explores TensorFlow but I don't personally see the pertinence to include code snippets when you'll learn how to use it in the programming assignments. 