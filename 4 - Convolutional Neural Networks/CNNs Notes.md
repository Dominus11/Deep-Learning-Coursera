## Helpful Resources

- [CS231n Course on CNNs](https://cs231n.github.io/convolutional-networks/)
- [Chapter 9 Convolutional Neural Networks of Deep Learning - Goodfellow, Bengio, Courville](https://www.deeplearningbook.org/)
- [3B1B's videos on Convolutions](https://www.3blue1brown.com/?v=convolutions)

# Week 1 - CNN Foundations

## Convolutions and Edge Detection

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

## Upgrades: Padding, Stride, Volume

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

## Convolutions in Practice

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
# Week 2 - Patterns in ConvNets
## Case Studies

### Classic Networks

#### LeNet-5

Original Paper: [[LeNet-5.pdf|Gradient-based learning applied to document recognition, LeCun et al. 1998]]

Observations:
- 60K parameters
- $\uparrow l \implies n_{H}, n_{W}\downarrow \land \; n_{C} \uparrow$
- Follows the common architecture of `(conv -> pool)^n -> fc -> fc -> output`
- At the time, people were using $\sigma, \tanh$ non-linearities, rather than ReLu.
- Whereas nowadays you would match your filters to have the same number of channels as the input, the original LeNet-5 has an unusual way of reducing the number of filters used by mapping which filters look at which block, due to low processing speed back then.
- LeNet-5 also had non-linearities after pooling
#### AlexNet

Original Paper: [[AlexNet.pdf|Krizhevsky et al., 2012. ImageNet classification with deep convolutional neural networks]]

Observations:
- Very similar to LeNet, simply deeper, had more parameters (60M parameters) and the vast ImageNet dataset.
- Used ReLU activations
- Trained on multiple GPUs
- Original AlexNet had a _Local Response Normalisation (LRN)_ layer, which normalises through channels. Not found to have a significant impact though.

#### VGG-16

Original Paper: [[VGG-16.pdf|Simonyan & Zisserman 2015. Very Deep Convolutional Neural Networks for Large-Scale Image Recognition]]

Observations:
- $f = 3, s = 1, p = \text{``same"}$ for about 16 layers, to give ~138M parameters
- Quite a simple architecture in following the `(conv -> pool)^n -> (fc)^m -> out` structure. Also very uniform due to the above hyperparameters. 
- $n_{c}$ kept doubling from 64 to about 512.
### ResNets (Fill in Intuition)

Original Paper: [[ResNet.pdf|He at al. 2015. Deep Residual Networks for Image Recognition]]

**Residual Block:** A residual block is a sequence of layers where you introduce _shortcuts/skip connections_ between layers. 

For the output of some layer $l$, $a^{[l]}$, you can then directly route this to layer $l+k, k \neq 1$, circumventing the _main path_ that the information usually takes. This will give:
$$a^{[l+k]} = g(z^{[l+k]}  + a^{[l]})$$

The effect we see in training is as follows:

```tikz
\usepackage{pgfplots}

\begin{document}
	\begin{tikzpicture}

		\begin{axis}[
			axis lines = left,
			xlabel = {\# Layers},
			ylabel = {Training Error},
			xmin = 0,
			ymin = 0,
			ymax = 1,
			xtick = \empty,
			ytick = \empty
		]
		
		\addplot [domain = 0:3, very thin, smooth, cyan] {e^(-x) + 0.1};
		\addplot [domain = 0:3, very thin, smooth] { 0.5*(x+0.7) + 1/(0.5*(x+0.7)) -1.6 };
		
		
		\end{axis}

	\end{tikzpicture}
\end{document}

```

The cyan line is our rough expectation of what should occur in training for a _plain_ network, but the white line is what actually occurs, simply because as you add more layers beyond a certain point, the network just gets harder to train for the optimisation algorithm. Instead, we see the cyan line occur with ResNets!! 

Andrew makes the argument that ResNets work because in a worst case, your network can simply learn the identity function from your previous inputs. I was somewhat unconvinced by that intuition.

### Inception Networks

#### 1x1 Convolutions and Bottleneck Layers

Original Paper: [[Network In Network.pdf|Lin et al. 2014 Network in Network]]

The idea of a $1 \times 1$ convolution may initially seem daft. Functionally, it might sound like you're scaling your input by a constant. However!! That would only be the case for a kernel of 1 channel. 

Instead, if you've got data comprised of multiple channels, then via the CoV operation, you can now reduce the dimension of your data to $n_{f}$ channels, when using $n_{f}$ filters. In other words, you can exploit the fact that CoV permits dimensionality reduction of data, which will allow you to save on computational costs.

We can apply this notion to yield _bottleneck layers_, which compress a high volume input into a lower volume representation, making it faster to compute on. Suppose you wanted to convolve on data of size $28 \times 28 \times 192$, with a $5 \times 5 \times 32$ kernel. You could:
- Do this directly, which would take $(28 \times 28 \times 32) \times (5 \times 5 \times 192) \approx 1.2 \times 10^8$ multiplications
- Apply a bottleneck! Compress the $192 \to 16$ first with a $1 \times 1$ convolution. Then apply your kernel. This will now take $(28 \times 28 \times 16)\times(192) + (28 \times 28 \times 32)\times (5 \times 5 \times 16) \approx 1.2 \times 10^7$ multiplications instead!!
#### Inception Nets

Original Paper: [[Going Deeper With Convolutions.pdf|Szegedy et al. 2014. Going Deeper With Convolutions]]

**Inception Module:** This allows you to combine the power of multiple layers into one, by using them each 'in parallel', and then concatenating the results to build an output comprised of many channels. 
- This leaves some edge cases whereby if you wanted to use a pooling layer, you would actually need to apply padding first.

An **Inception Net** is an architecture which:
- Makes use of many of the aforementioned inception modules
- Has a slightly less linear structure which permits for side branches that can yield 'early outputs'. This has something of a regularising effect by making sure that the early outputs aren't too shoddy themselves
### Edge Computing

MobileNet V1: [[MobileNets.pdf|Howard et al., 2017, MobileNets: Efficient CNNs for Mobile Vision Applications]]
MobileNet V2: [[MobileNet V2.pdf|Sandler et al., 2019, Inverted Residuals and Linear Bottlenecks]]
EfficientNet: [[EfficientNet.pdf|Tan et Le, 2019, EfficientNet: Rethinking Model Scaling For Convolutional Neural Networks]]
#### Depth-wise Separable Convolutions

**Goal:** _Accelerate inference speeds_, by improving upon our existing convolution algorithm. This is for applications in edge computing, rather than deferring computation to centralised computers.

We shall do so by now _approximating/altering the convolution operator_ into something slightly new and much faster. We will divide this new convolution stage into two phases:
- **Depthwise Convolution:** This is where we take a regular image $I: n \times n \times n_{c}$, and apply $n_c$ filters $F_{i}: f \times f$ to each layer in a planar fashion to produce output $X: n_{out} \times n_{out} \times n_{c}$. This performs mixing of features within each layer. 
  
- **Pointwise Convolution:** We then take $X$ as our new input, and we apply $n_{f}$ pointwise filters $P_i: 1 \times 1 \times n_{c}$ under the CoV operation, to yield our final output $O: n_{out} \times n_{out} \times n_{f}$. This mixes the extracted features between layers.

We end up finding that now inference is $\frac{1}{n_{c}} + \frac{1}{f^2}$ faster than previously.

#### MobileNet

Let `DSC` denote the depthwise separable convolution operation. 

- **MobileNet v1:** `(DSC)^13 -> Pool -> FC -> Softmax`
- **MobileNet v2:** `(Bottleneck Block)^17 -> Pool -> FC -> Softmax `
	Residual Connection between `N1, N2`. 
	Bottleneck block: `N1 -> Expansion -> Depthwise -> Projection -> N2` + Residual(`N1, N2`). 

Let's talk about the upgrade that arrived with V2 especially. The _bottleneck block_ achieves two things:
- The expansion operation increases the size of representation, learning a richer function
- When deploying on a mobile device, the projection operation makes the representation smaller again, reducing memory requirements on a constrained device.

#### EfficientNet

If you want to make a CNN more effective, they observe that there are 3 factors to help improve it:
- $r$ - Image Resolution
- $d$ - Neural Met's depth
- $w$ - Width of the layers in the neural net.

Given a certain set of constraints on computational resources depending on the device, the question then becomes: 'How can we rescale these parameters to achieve the best performance?' Your answer can often be found by looking at open source implementations.
## Application

### Transfer Learning

Often it's the case that you can make effective use of someone else's open source model that is pre-trained already, on a similar task, and then apply [[Structuring ML Projects Notes#Transfer Learning|transfer learning]], with a few tricks:
-   If you choose to undertake a pre-training regime, what you could do is put all of your training data through all layers except the final ones, _pre-computing the feature vectors and saving them to disk_ to accelerate training, since now you're effectively only training a softmax layer (in a classification context). 
- If you've got lots of data, then you could choose to undertake a fine-tuning regime where you:
	- _Freeze fewer layers, or even replace some outright_. In particular, if you have a high volume of data, then you can actually ply the network with more layers on top, effectively training an additional neural network that learns the mapping from feature vectors to outputs.
	- _Use the entire network as initialisation_ and then train on top of that to adjust the weights to get to the mapping you need.

### Data Augmentation

Data Augmentation solves the problem of not necessarily having enough training data to achieve your task. It generally helps you to abstract the learned representation by eliminating introducing invariants into the data. You can try:
- **Spatial/Geometric Transformations** 
- **Random Cropping**
- **Colour Shifting** 

**Q:** Give some examples of spatial/geometric transformations. 
	**A:** Mirroring, Shearing, Rotations, Local Warping

**Q:** How do you generally pick the colours used in colour shifting?
	**A:** You draw them from some random distribution of your choosing. 
	An extension is _PCA colour augmentation_, described in the AlexNet paper, which tries to preserve the main colours of your image by using PCA to extract what the main colour components are, and then weighting these higher when deriving the colour shifts to be used.

**Q:** What's the motivation of using colour shifting?
	**A:** This is to introduce invariance under different lighting conditions, which might give rise to slightly different colours that ultimately don't affect the property you're trying to observe.

You may also wish to parallelise the processes of loading/augmenting your data and training on it.

### General Advice

It is a common pattern that the less data you have, the more hand-engineering you need to do, and this requires a fair bit of insight to do successfully.

To succeed on benchmarks and win competitions, you might want to try:
- **Ensembling:** Train multiple networks in tandem, and then average their outputs. 
- **Multi-Crop At Test Time:**
	- This effectively uses our data augmentation techniques in order to give us a larger set of test data, by producing multiple copies of an image, and then taking various crops from each copy. You can then average the results across each crop to give your final result for that image.

You'll generally want to use existing architectures that exist in literature, especially open source implementations. And don't underestimate the power of fine-tuning!!
# Week 3 - Detection Algorithms

[YoLo v7 Detection Algorithm](https://www.v7labs.com/blog/yolo-object-detection)

# Week 4 - Face Recognition + Neural Style Transfer