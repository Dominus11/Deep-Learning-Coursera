
# Week 1 - RNNs

## Introduction + Concepts

**Prompt:** Give some examples of supervised learning tasks involving sequence data!
	- Speech Recognition
	- Music Generation
	- Sentiment Classification
	- Machine Translation
	- DNA Sequencing
	- Video activity recognition
	- Name entity recognition

**Notation:** Let $x$ be a data sequence whose elements belong to some discrete vocabulary/domain $V$. 
- $x^{\langle t \rangle}$ indicates the $t$-th item in $x$
- $T_x$ denotes the length of $x$
- You can represent $x^{\langle t \rangle}$ with a _1-hot encoding_, augmented with an `<unknown>` token.

### RNNs

**Q:** Why is a standard neural-net architecture ineffective? Also state a practical limitation of such a model of computation.
	- Inputs and outputs don't share lengths across examples
	- Such an architecture won't share features learned across multiple positions of text.
	- Any input layer is going to be _huge_, on the order of $O(m|V|)$ for $m$ training examples, $|V|$ being the size of the vocabulary. We'd also want to change our representation to reduce the number of parameters.

**Insert Tikz Diagram Here...**

The forward propagation step for an RNN cell at timestep $t \geq 1, t \in \mathbb{N}$ is given by:

$$
\newcommand{\time}[1]{{\langle #1 \rangle}}
\begin{align*}
a^{\time{t}} &= g_{a}(W_{aa}a^{\time{t-1}} + W_{ax}x^{\time{t}} + b_{a}) \\
\hat{y}^{\time{t}} &= g_{y}(W_{ya}a^{\time{t}} + b_{y}) \\
\end{align*}
$$

Where we usually have $a^{\time{0}} = \mathbf{0}$.  

**Notation:**
	- $W_{xy}$ should be read as a weight of a $y$-like variable contributing to an $x$-like variable. 
	- $b_x$ should be read as a bias contributing to an $x$-like variable.
	- $g_{x}$ should be read as an activation function producing an $x$-like variable.


We observe that these formulae are slightly bulky, so we're going to add some syntactic sugar and spice. The new equations will be given:

$$
\begin{align*}
a^\time{t} &= g_{a}(W_{a}[a^\time{t-1} , x^\time{t}] + b_{a}) \\
\hat{y}^\time{t} &= g_{y}(W_{y}a^\time{t} + b_{y})
\end{align*}
$$
Here, we have:
$$
\begin{align*}
W_{a} &= 
\begin{bmatrix}  
 & W_{aa} &| & W_{ax} & \\
\end{bmatrix}  \\ \\
[a^\time{t-1}, x^\time{t}] &= 
\begin{bmatrix}
 a^\time{t-1} \\ 
 x^\time{t} \\
\end{bmatrix}
\end{align*}
$$

In each case these denote concatenations of the matrices/vectors in question, and you can verify that this would yield the same result.

For backward propagation, we can perform a process known as _backpropagation through time_. It earns this name because if we unfold the computation graph of an RNN cell on some sequence $x$, you effectively get a neural network of depth $T_{x}$, which emits a vector $y^{\time{t}}$
at each timestep/layer, and therefore as the backprop occurs, weights get shifted back through the time-indexed layers.

**Q:** Construct an appropriate cost function for this architecture, given an arbitrary loss function $\mathcal{L}$. 
	**A:** First, at each timestep, take the individual loss, $\mathcal{L}$ between $y^{\time{t}}, \hat{y}^{\time{t}}$. Then, add up these losses for all values of $t$, giving: $$ \mathcal{C} = \sum_{t=1}^{T_{x}} \mathcal{L}(y^{\time{t}}, \hat{y}^{\time{t}})$$
	This allows for you to consider the individual loss at each timestep.

**Q:** What's a problem with _unidirectional_ RNNs?
	**A:** At a given point $t$ in a sequence, you don't make use of any data further along in the sequences, meaning you lose extra context.

You can categorise most RNN architectures in terms of the cardinalities of their outputs:
- **1-To-1**
- **1-To-Many:** Pass in an input value, generate successive outputs and pass them into the next phase - i.e. ($\hat{{y}}^\time{t} = a^\time{t}$)  - until termination. 
- **Many-To-1:** Read in an input sequence $x$ and then after reading $x^\time{T_{x}}$, output a value
- **Many-To-Many (M2M):** Read in an input sequence and yield an output sequence.

**Q:** Describe two types of M2M architecture. 
	- Only if $T_{x} = T_{y}$, you can read in $x$ items and at each stage output an item of $y$. (The vanilla architecture described above).
	- If $T_{x} \neq T_{y}$, then you can have an _encoder_ which reads in all of $x$ and a decoder which takes that context and produces all of $y$. Salient to machine translation. 

**Q:** Describe the problems that one would expect to arise in an RNN? Which is the more subtle problem?
	- For domains like NLP where there can be many long-term dependencies, you can experience gradient issues. 
	- Exploding gradients are far simpler to detect, since you will be terrorised by `NaN` values. 
	- Vanishing gradients are more subtle and are indicative of a lack of context 'memory' with increasing length dependencies.

### NLP: Language Models

**Goal:** To construct a model that allows us to determine the most likely types of sentences that can emerge in a language, and generate them.

We're going to train on a very large and dense _tokenised_ corpus of text. 

To train, we feed in tokens of text, and the RNN at each timestep will yield the output of a softmax layer - a likelihood distribution of each token appearing, given the tokens that have been seen so far. 

**Q:** What is the basis for this recurrence? (I.e. what do you feed it at $t = 1$).
	**A:** $x^\time{1} = a^\time{1} = \mathbf{0}$. This means that we start it off having seen no tokens and, as before, with no real activation.

To sample from such a model, we can run the model over multiple timesteps, and at each timestep, randomly sample a word from the generated distribution.

**Q:** What are two important tokens that could be helpful/necessary? How could you make do without them?
	- `<UNK>` for unknown symbols that are in your corpus but not your pre-established vocabulary. You can make do without this by making a _character-level model_ where you sample characters to termination, which also allows for any possible sequence of words.
	- `<EOS>` for the end of string. You can generate sequences without this by sampling for a fixed number of tokens, rather than sampling until `<EOS>` is generated.
**Q:** What are the appropriate loss/cost functions?
	**A:** Softmax loss and then the cost should simply be the sum of the losses over all timesteps.


## Advanced Architectures

### GRUs

**Goal:** Improve the retention of long-term dependencies and mitigate vanishing gradients.
**Papers:**
- [[Empirical Evaluation of GRUs on Sequence Modelling.pdf]]
- [[On the Properties of Neural Machine Translation.pdf]]

The GRU - _Gated Recurrent Unit_ - attempts to add an explicit memory mechanism to the RNN cell, aside from simply praying that the weights will tweak themselves such as to allow important values to propagate through the activations. 

We shall tweak the system s.t. $a^\time{t} := c^\time{t}$, the value of the _memory cell_ at time $t$. The update mechanism for a GRU will be as follows:

$$
\begin{align*}
\Gamma_{r} &= \sigma(W_{r}[c^\time{t-1}, x^\time{t}] + b_{r}) \\
\tilde{c}^\time{t} &= \tanh(W_{c}[\Gamma_{r}\ast c^\time{t-1}, x^\time{t}] + b_{c}) \\
\Gamma_{u} &= \sigma(W_{u}[c^\time{t-1}, x^\time{t}]+b_{u}) \\
c^\time{t} &= \Gamma_{u}\ast\tilde{c}^\time{t} + (1-\Gamma_{u}) \ast c^\time{t-1}
\end{align*}
$$

**Q:** Translate this scheme into English. What does the cell do, and what do the variables represent?
	- $\Gamma$ variables represent _gate_ values, 'bit vectors' corresponding to whether an entry in the memory cell shall pass onto the next stage or not. $c$ variables represent _memory vectors_, with $\tilde{c}$ representing the _candidate cell values_ for the next time step.
	- $\Gamma_{r}$ is a _relevance gate_ which takes in the previous memory and current input token and determines how significant the memory values are to propagate along. In an ideal world, the values are binary, but in practice they are given as the output of a softmax function.
	- $\tilde{c}$ is the vector of candidate values for the cells at this timestep, which weights the old memory in terms of its relevance against the new token.
	- $\Gamma_{u}$ is an _update gate_, which will determine whether the old memory will persist or if it will be supplanted by the new candidate memory
	- If $\Gamma_{u} \approx 1$, then the new candidate memory is more strongly weighted in the production of $c^\time{t}$, otherwise the memory from the previous timestep is more strongly weighted.

**Q:** How could we simplify the GRU to prevent vanishing gradients?
	**A:** Increase the sensitivity to $c^\time{t-1}$ at each timestep by setting $\Gamma_{u} = \mathbf{1}$. 


### LSTM Units

**Goal:** Improve the retention of long-term dependencies and mitigate vanishing gradients.
Papers: [[LSTMs.pdf]]

In the GRU, our update was given by:

$$c^\time{t} = \Gamma_{u}\ast\tilde{c}^\time{t} + (1-\Gamma_{u}) \ast c^\time{t-1}$$
**Q:** What does this mean in theory for the memory retention capacity if $\Gamma_u \in \{0,1\}$? 
	**A:** You necessarily have to forget either the memory you were going to commit to storing or some old memory from before.

The LSTM has separate memory, activation and token inputs, and these 3 gates:
- $\Gamma_{f}$ - The _Forgetting Gate_. This replaces the $(1-\Gamma_{u})$ term in the above update rule, enabling a higher degree of flexibility in what you forget.
- $\Gamma_{u}$ - The _Update Gate_, as before. 
- $\Gamma_{o}$ - The _Output Gate_

The update rules for an LSTM are presented as follows:

$$
\begin{align*}
\tilde{c}^\time{t} &= \tanh(W_{c}[a^\time{t-1}, x^\time{t}] + b_{c})\\
\Gamma_{i} &= \sigma(W_{i}[a^\time{t-1}, x^\time{t}] + b_{i}) \;, i \in \{o,f,u\} \\
c^\time{t} &= \Gamma_{u} \ast \tilde{c} + \Gamma_{f} \ast c^\time{t-1} \\
a^\time{t} &= \Gamma_{o} \ast (\tanh c_{t})
\end{align*}
$$

**Q:** Identify a necessary condition on the relative dimensions of $\Gamma$ and $c$ vectors.
	**A:** They must be the same to support an element-wise product. 
**Q:** Describe a variation on the calculation of the gate values.
	**A:** _Peephole connections_ also allow you to determine the gate values in terms of the previous memory. When using this, we also have that by definition, $c^\time{t-1}_{i}$ can only affect $c^\time{t}_{i}$. 
**Q:** Describe how the LSTM enables effective long-term memory.
	**A:** The separate memory line makes it really easy for early $c$ values to just be propagated along without any obstruction.

### Bidirectional RNNs

**Goal:** To enable bidirectional context flows.

This can be achieved by dividing forward propagation into two parallel streams, defining an acyclic graph:
- _Context Forward:_ This generates context-forwarding activations, $\overrightarrow{a}^\time{t}$, in the usual way.
- _Context Backward:_ This generates activations transmitting data backwards, $\overleftarrow{a}^\time{t}$, starting at the end of the input sequence and going backwards through in the usual fashion. 

This gives us the update rule:
$$\hat{y}^\time{t} = g(W_{y}[\overleftarrow{a}^\time{t} , \overrightarrow{a}^\time{t}] + b_{y})$$

**Q:** What's a disadvantage of a BRNN?
	**A:** You can't lazy load sequences, you need the entire thing to start off, which might make inference slower in something such as real-time speech applications.
### Deep RNNs

**NB:** Up to this point, we've only actually considered how an individual _unit_ unfolds across the time dimension to make a mini-network. However, we still need to compose them to make _deep RNNs_. 

**Q:** What depth $L$ constitutes a 'deep' RNN?
	**A:** $L \geq 3$ . Any further than this and you increase the surface area for vanishing gradients to crop up.

To compose multiple RNN layers together, you can consider the process as a pipeline with 2-dimensions:
- In the _temporal dimension_, hidden layer outputs recur back into the cells that produced them to influence future computation. 
- In the _layer dimension_, hidden layer outputs also propagate through the layer stack, in a fashion very much similar to an instruction pipeline, going from layer to layer until reaching the output. In this dimension, you can almost imagine layers lazy-loading from the preceding layer, and spitting out a sequence at the end point.

(This is to be assisted with a TikZ diagram at some point I hope). 

# Week 2 - NLP, Word Embeddings




# Week 3 - Sequence Models and Attention Mechanisms



# Week 4 - Transformers