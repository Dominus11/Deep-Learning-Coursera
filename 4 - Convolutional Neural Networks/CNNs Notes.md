
## Week 1 - CNNs 

### Convolutions and Edge Detection

**NB:** You may wish to watch [3B1B's videos on Convolutions](https://www.3blue1brown.com/?v=convolutions)!! They're very informative in building up the idea of what a convolution is. 

**Convolution:** In digital signal processing, an operation on a signal, which allows us to identify features in that signal, by sliding a _kernel_ over your image to yield a _feature map_. Technically speaking, the operation we will see is _cross-correlation_.

We denote the operation as follows:
$$I * K = F$$
We denote:
- $I$ - The image. For now, we will assume this is a matrix but in practice it's a tensor with a depth of 3 for each of the RGB colour channels.
- $K$ - The Kernel. This is a matrix which we will slide over our image. For each pixel, we will perform an element-wise product between $K$ and the items in the image which are covered by $K$, and then add these up.
- $F$ - The _Feature Map_. This is the output matrix/image containing the results of the convolution at each cell.

The image processing task we wish to perform is influenced by the kernel we select. For example, one important task we might wish to perform is _edge detection_, which we can achieve with the kernel:

$$
K = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1 \\
\end{bmatrix}
$$

**Q:** What does a positive number being output from this convolution mean?
	**A:** A transition from light to dark going left to right, since if you've got high numbers on one side and low numbers on the right, these combine to give a resulting positive number. In other words, this achieves vertical edge detection.
**Q:** How could you achieve horizontal edge detection?
	**A:** $K \to K^T$. Swap out the kernel for its transpose.
**Q:** What's the problem with this filter?
	**A**: The problem with this filter is that it gives equal weightings to values further away (the diagonal cells). So you might want to change the exact filter used by making the difference between the corner cells and outer middle cells more stark, such as with _Sobel/Scharr_ filters. Better yet, _let the kernel's entries be parameters to learn_, which can detect edges at the orientation that allows for the most information to be extracted!!

### Upgrades: Padding, Stride, Volume

You also run into some problems with the convolution operation:
- [c] _Shrinking Output:_ $F$ shrinks in dimension relative to $I$. Given an $n \times n$ image and a $k \times k$ kernel, then $F$ has dimension $(n-k+1) \times (n-k+1)$. So you can't repeat this convolution process forever. 
- [c] _Edge Information Loss:_ You can only look at cells where the kernel is fully on the image. This means that you lose edge information because you can't give those cells their own convolution.

This motivates **padding**. You pad the image with 0s by a certain depth, which we shall conventionally refer to as $p$. $p = 1$ is sufficient for removing both of the aforementioned problems. There are two commonplace options for choosing a padding depth, which are:
- _Valid Convolutions:_ Just don't pad. 
- _Same Convolutions:_ Pad s.t. $\dim I = \dim F$. I.e. s.t. $n + 2p - k + 1 = n$. 

**Q:** Why do you usually have that $k$, the kernel size is odd?
	**A:** If it's even, then there is no 'centre'. If it's odd, you actually have a centre to focus the kernel on. This is helpful for same convolutions because the formula for selecting padding size also only permits odd kernel sizes.

When scanning over the image, you may also wish to take steps larger than that which moves you onto the adjacent cell/row. The step size you take is referred to as the **stride**, $s$.  

**Q:** What is the dimension of the feature map in terms with a stride of $s$?
	**A:** $(\frac{n + 2p -k}{s} + 1) \times \frac{n + 2p -k}{s} + 1)$. We should note that we actually use the floor of the fraction instead, but I wanted the notes to look somewhat pretty.

We simplified a practicality earlier that images do not have 3 colour channels. In actuality, we must perform _convolutions over volume_. To do so, perform convolutions with kernels for each colour channel. These will each yield scalars. Then add these together to yield your final result for that pixel in the image. 

**Q:** Deduce the dimensionality of the CoV operation.
	$$(n \times n \times n_{c}) \;\;* (k \times k \times n_{c}) \to (n-k+1) \times (n-k+1) $$
	In essence, you are reducing volumetric data to planar data, which might not be the intuitive result.

Another thing you can do is apply multiply filters at once. For example, performing edge detection at two angles! In this case, performing multiple convolutions over volume gives multiple matrices as your result, which can then be layered to make a tensor.

**Q:** Deduce the dimensionality of the upgraded CoV operation.
	$$(n \times n \times n_{c}) \;\;* \;\;(k \times k \times n_{c})^{n_{f}} \to (n-k+1) \times (n-k+1) \times n_{f}$$
	The exponential notation on the kernel indicates repeating the operation $n_f$ times in parallel. Here, you have made volumetric data by repeatedly generating planar data. 

### Convolutions in Practice

#### Convolutional Layers

We must now understand how we can use all that we've built up in practice. The convolutional layer (suppose it is layer $l$ in our network) will work as follows:

1. Into layer $l$, input $I: n_{H}^{[l-1]} \times n_{W}^{[l-1]} \times n_{c}^{[l-1]}$. 
2. Perform any necessary padding on $I$, by padding amount $p^{[l]}$. 
3. Given $n_{c}^{[l]}$ filters  $F_i$ of dimension $f^{[l]} \times f^{[l]} \times n_{c}^{[l-1]}$, compute $C_{i} = I * F_{i}$, where $*$ denotes the CoV operation. $C_{i}: n_{H}^{[l]} \times n_{W}^{[l]}$. 
4. Then compute $z_{i} = C_{i} + b_{i}$, where $b_{i} \in \mathbb{R}$, and we assume broadcasting notation.
5. Layer the $z_i$s together to make the tensor $z: n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$. 
6. Compute $a = g(z)$, where $g$ is the activation function for the layer.  

**NB:** My description makes it seem as though you sequentially perform the convolution for each filter, but in practice you can actually do these in parallel, since convolution is a SIMD-style operation. It just phrased like this to help you conceptualise it.

You can also parallelise this across all $m$ training examples to yield the final output tensor $A: m \times n_{H}^{[l]} \times n_{W}^{[l]} \times n_{c}^{[l]}$. 

#### Pooling Layers

**Pooling:** An operation where you pass another filter over an input to try and extract/preserve important features.

A pooling layer only has the hyperparameters $f$ and $s$, and no actual parameters to learn via gradient descent. You can _sometimes_ use padding. Another hyperparameter is the type of pooling you use.

_Max Pooling_ is a commonly effective implementation of pooling, which works by checking for the existence of important features (denoted by a high number) and then propagating them through to the next layer via the $\max$ operation. If the numbers are low in some subregion of the input, then the result of the pooling layer for that region will stay low, pointing out that this is insignificant by comparison.

_Average Pooling_ takes the average of the pixels in the subregion. You might use this to collapse your representation into a slightly smaller state, such as $7 \times 7 \times 1000 \to 1 \times 1 \times 1000$. 

You can also parallelise these with multiple channels in a similar way to convolutional layers.
#### ConvNets

Naming Conventions:
- You can call refer to conv/pooling layers as 2 separate layers.
- You can also call a conv layer, followed by a pooling layer a full layer, since the pooling layer has no parameters of its own, so it arguably doesn't exist. Conventionally, researchers report the number of layers with parameters. Hence, the latter is Andrew's convention.

**Fully Connected (FC) Layer:** After a layer, you can flatten the output into a vector. From here, you can just pass it into a layer that you would see in a regular ANN, which we refer to as a fully connected layer.

We like ConvNets because of
- [u] _Parameter Sharing:_ Where the feature detectors/filters can be applied across multiple parts of the image, reducing parameters in this way, and thus tendency to overfit. 
- [u] _Sparsity of Connections:_ In each layer, the output values depend on a small number of inputs, which also reduces proneness to overfitting. It also allows us to use smaller training sets, reducing the number of parameters even further.
- [u] Convolutional structures also help boost _translational invariance_, since you're learning the filters from allll across the image rather than just specific regions.
## Week 2 - Case Studies + Using ConvNets



## Week 3 - Detection Algorithms

[YoLo v7 Detection Algorithm](https://www.v7labs.com/blog/yolo-object-detection)

## Week 4 - Face Recognition + Neural Style Transfer