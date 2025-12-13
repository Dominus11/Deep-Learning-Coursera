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
$$\newcommand{\time}[1]{{\langle #1 \rangle}}$$
$$
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

**Prompt:** Up to this point, we've only actually considered how an individual _unit_ unfolds across the time dimension to make a mini-network. However, we still need to compose them to make _deep RNNs_. 

**Q:** What depth $L$ constitutes a 'deep' RNN?
	**A:** $L \geq 3$ . Any further than this and you increase the surface area for vanishing gradients to crop up.

To compose multiple RNN layers together, you can consider the process as a pipeline with 2-dimensions:
- In the _temporal dimension_, hidden layer outputs recur back into the cells that produced them to influence future computation. 
- In the _layer dimension_, hidden layer outputs also propagate through the layer stack, in a fashion very much similar to an instruction pipeline, going from layer to layer until reaching the output. In this dimension, you can almost imagine layers lazy-loading from the preceding layer, and spitting out a sequence at the end point.

(This is to be assisted with a TikZ diagram at some point I hope). 

# Week 2 - NLP, Word Embeddings

## Introduction

**Prompt:** What is the issue with the one-hot encoding we have been considering up to this point?
	**A:** By all reasonable measures (inner product, distance), all words are entirely independent. In other words, they express the unique identity of a word, but are entirely unexpressive as to its underlying potential semantics.

With this in mind, we instead want to learn featurised representations_ of the words in our vocabulary. The underlying principle:
- Choose a suitable dimension, $d$, with which to express our vocabulary.
- Learn basis vectors spanning that space, that each represent some non-obvious semantic construct.
- Learn the coefficients of our words with respect to each basis vector. 

This should be reminiscent of our [[CNNs Notes#Siamese Networks|Siamese Networks]], where we learned an $f_{\theta}: I \to \mathbb{R}^d$ for some arbitrary $d$, an encoding of the image. 

When representing word embeddings for $d \geq 3$, we use algorithms like t-SNE which remove the parallelogram similarity/analogy notion

**Q:** How does this enable us to learn analogies?

**Q:** Motivate two possible similarity metrics under this paradigm

**Q:** Why are they more effective than one-hot encoding?
	**A:** Algorithms can generalise better and learn from less training data, since you're now providing extra semantic information to the model for free.

**Q:** How many one apply transfer learning to word embeddings?
	**A:** Find a set of word embeddings online, and then fine-tune them on your own smaller corpus.


## Learning Embeddings

### General Algorithm

**Embedding Matrix:** A matrix $E$ of dimension $d \times |V|$, linearly mapping our one hot vectors to their respective embeddings.

**Q:** What are $d$ and $|V|$?
	- $d$ : The _dimensionality_ of our embedding vectors.
	- $|V|$: The number of words in our vocabulary

The general method of learning embeddings is as such:
- Take in a sequence of tokens $X$. 
- For each _target_ token, $T$, in the sequence, identify a _context_, $C$, a set of tokens you want to use as features for that target. 
- Generate the probability distribution for what $T$ is, and fit it using MLE (via a softmax layer and the log-loss function)

The structure of the neural network you get will be:
- _Embedding Layer_, which contains the embedding matrix, trained to produce embedding vectors which are most effective in densely capturing semantic meaning. 
- _Softmax Layer_, which takes in the embedded contexts and trains its own weights and biases to shift the probability distribution towards predicting the true target.  

**Q:** You can generalise this algorithm by broadening the context set you select. Give examples of such sets. 
	- _Fixed Window:_ Select $l$ tokens before the target and $r$ tokens after the target. 
	- _Markovian:_ $l = 1, r = 0$. Presume that the target only depends on the previous token. 
	- 1 nearby word: Randomly sample from within a fixed window.


### Word2Vec: Skip-Gram

Original Paper: [[Mikolov et al. Efficient Estimation of Word Representations In Vector Space.pdf]]
$\newcommand{softmax}{\operatorname{softmax}}$
**Skip-gram Model:** For some arbitrarily sampled context words in the corpus, define a fixed window around it, and try to determine target words in that window around it. 

**Q:** What is the goal of the skip-gram model?
	**A:** The skip-gram model is defined to try and find surrounding words of a certain input word.

**Architecture:**
- Input: One-hot encoding of the context token, $o_{c}$
- Take this and multiply it by $E$ to produce the embedding vector $e_c$. 
- Pass this into a $\softmax$ layer, which estimates the categorical variable $(t|c), t \in V$.  
- Compare to log-likelihood loss and back-propagate. 
- Export the $E$ matrix when done training. 

The conditional distribution for the target words given the context is given by:
$\newcommand{\dotprod}[2]{\langle #1 , #2 \rangle}$
$$\Pr(t|c) = \frac{e^{\dotprod{\theta_{t}}{e_{c}} }}{\sum_{w=1}^{|V|} \exp\left( \dotprod{\theta_{w}}{e_{c}} \right)}$$

Where $\theta_{w}$ is a vector parameter for each word in $V$, given by the corresponding row in the weights matrix, $W$, of the $\softmax$ layer. In other words, $\theta_{t} = W_{t\_}$. 

**Q:** Identify an inefficiency in computing this formula. How can you fix this?
	**A:** It's incredibly slow to compute a loop on the order of $|V|$ for _every_ inference, which is usually $\sim 10^5 - 10^7$. We can fix this by using a hierarchical classifier, by using a binary/Huffman tree as the classification architecture, to enable logarithmic computation speeds. 

**Q:** How do you sample the context? 
	Use a heuristic to warp the distribution slightly to your needs, such as removing stop-words. 

**Q:** Both $E$ and $W$ store embeddings for the words in $V$. What properties do the embeddings from each matrix have? 
	**A:** $E$'s embeddings represent the active semantics of the words - how it affects the interpretation of the words around it. $W$'s embeddings are more oriented to do with classification, since each row is tuned to accurately shift the probability distribution yielded by the $\softmax$ layer.
### Skip-Gram Enhancement: Negative Sampling

[[Mikolov et al. Distributed Representations of Words and Phrases and their Compositionality.pdf]]

**Goal:** To more efficiently learn embeddings in comparison to the current skip-gram architecture, by setting up a new, but equivalent, learning problem. 

**Negative Sampling:** Given a context token and another token, determine if the token is a target? Pick one true example as-per the skip-gram sampling model and then randomly sample $k$ false examples from $V$, the dictionary.

**Q:** Give the form of the function learned, from a probabilistic ML perspective 
	$$f: C \times V^{k+1} \to [0,1] $$
	The model will learn the probability that for each $(c,t_{i})$ pair out of the $k+1$ target samples, $t_{i}$ is indeed a target for $c$. We could argue that this is learning a boolean, but it's easier to make the probabilistic argument.  

In essence, you treat the old problem as a _logistic regression_ problem, replacing the softmax layer with a sigmoid layer. 

**Q:** Describe the new training procedure:
	- Iterating over each sampled context word. 
	- For each of the samples (positive and negative alike) for that context word, take the output of the sigmoid layer and back-propagate using the log-loss.

**Q:** Why is this substantially more efficient than the previous skip-gram model?
	**A:** Previously, you had to do 10000 binary classification problems each time. Now you only update $k+1$ of them on every training iteration, which is far more efficient in weight updates.

**Q:** How could you sample context words? 
	- According to empirical frequency in corpus, however this faces the weighting problem like the accuracy metric. You may also want to remove stop-words again. 
	- You could also assume word appear as a uniform random variable. However, this is non-representative of the observed distribution in language. 
	- Generally, take a heuristic value between the two. For example, Mikolov et al. chose: $$P(C = w_{i}) = \frac{f(w_{i})^{0.75}}{\sum_{j}f(w_{j})^{0.75}}$$
	  Where $C$ is a categorical random variable whose domain is $V$ for the context variable. 
### GloVe 

Original Paper: [[Pennington et al. GloVe- Global Vectors for Word Representation.pdf]]

**Goal:** Abandon sampling and learn directly from the statistical co-occurrence properties of a given corpus, as a direct regression problem.

$$\sum^{|V|}_{i,j = 1,1} f(X_{ij})(\theta_{j}^Te_{i} + b_{j} + b_{i}- \log X_{ij})^2$$
Where:
- $X_{ij}$ is the frequency of word $i$ being the context of word $j$, where $w_i,w_j \in V$. 
- $f$ is a weighting function
- $b_i, b_{j}$ are biases corresponding to the given words: $w_i, w_j$. 

**Q:** What properties should $f$ have?
	- Should be 0 for $X_{ij} = 0$, such as to not train from unseen instances.
	- Non-decreasing, such as to not over-optimise for infrequent relations.
	- Shouldn't overly weight stop words.

**Q:** Why might you choose to average the $\theta_w, e_w$ vectors at the end of training?
	**A:** Since $\theta_w, e_w$ are symmetric in the formula, this means that when training occurs, semantic information gets smattered across them each. So in order to restore all the semantic information, you take the linear average of the two (this is a linear model and so it makes sense to take a linear combination).

## Applying Word Embeddings

### Sentiment Classification

**Q:** Present 2 methods to perform sentiment classification with word embeddings.
	- Average the word embeddings of all the tokens in the text, and take the sentiment of that, using a softmax classifier (for more expressive ratings, such as stars or positive/negative). Not terrible, but things can cancel out or omit negations. 
	- You can upgrade this to an RNN, now taking context/full structure into account, to prevent the issue that something like Naive Bayes runs into, where you accumulate 'positive' embeddings together.

### Debiasing Embeddings

Original Paper: [[Bolukbasi et al. Man is to Computer Programmer, as Woman is to Homemaker.pdf]]

One school of thought on this is to:
1. Identify the bias direction
2. Where words aren't definitionally 'biased', project their embeddings onto the non-bias subspace. 
3. Equalise pairs - make them equidistant from/symmetric about the bias axis.


**Q:** How do you identify the bias direction?
	**A:** Average the difference between the embeddings for a few pairs of words which are definitionally biased in opposite ways. E.g. for gender, take the differences between ('he', 'she'), ('him', 'her'), ('man', 'woman'), etc and average those. 
**Q:** Why would you want to equalise biased pairs?
	**A:** You want all unrelated words to be equidistant from the bias axis, since this will remove any underlying semantic bias, so that they measure as having the same similarity with either direction along the biased axis. 
**Q:** How do you work out what words to normalise/equalise?
	**A:** You can try to use a classifier architecture which can tell the difference.
# Week 3 - Sequence Models and Attention Mechanisms

### Sequence Models

Some candidates domains for this task include:
- _Machine Translation:_ For a given input sequence in language A, output a new sequence in language B. 
- _Image Captioning:_ Not quite a sequence, but you can input an image, convert it into an embedding via a pre-trained network, and then feed this embedding to an RNN to make it generate a caption

**Q:** What is the general architecture for solving these tasks?
	**A:** An encoder-decoder network, with one half encoding your input into a dense embedding, and the other decoding it back out into a sequence.
**Q:** What is a nuance in how the RNN generates the output sequences for these, in comparison to automatic sequence generation?
	**A:** You now want to generate the most likely sequence, rather than randomly sampling a language model/distribution, since random sampling can give a bad translation.
**Q:** Frame machine translation as a probabilistic learning task.
	**A:** Given an input sequence $a^\time{1}, \dots, a^\time{T_{a}}$ of tokens in language $A$, you now want to find the sequence $b^\time{1}, \dots, b^\time{T_{b}}$ in language $B$ which maximises $\Pr(b^\time{1}, \dots, b^\time{T_{b}} | a^\time{1}, \dots, a^\time{T_{a}})$.  
#### Beam Search

**Goal:** Find the most likely output from a conditional language model, using an approximate searching method. It's like a more selective breadth first search.

**Intuition:** Keep branching out and then prune the tree to only include the $B$ most likely sequences, and evaluate the sentence fragments using $B$ parallel, distinct instances of the decoder. 

The algorithm will yield at each timestep:
$$
\begin{align*}
\operatorname{arg-B-}\max_{y} &\Pr(y^\time{1}|x)&& \text{[Basis]}\\
\operatorname{arg-B-}\max_{y} &\prod_{t=1}^{T_{y}} \Pr(y^\time{t} | x, y^\time{1}, \dots, y^\time{t-1})&& \text{[Recursion]}
\end{align*}
$$

Here, $\operatorname{arg-B-}\max$ denotes that at each iteration/timestep, you return the $B$ arguments which maximise the given function. In this case, at each timestep, output the $B$ sequences (we're maximising over the possible $y$-s, not the tokens) which maximise the likelihood of seeing the new token, conditional on all previous tokens and the input sequence.

**Q:** What is a beam search with $B = 1$ also known as? Why is it unsuitable?
	**A:** A beam search with $B =1$ is a _greedy search_, where at each stage you select the most likely current sequence of words. This is unsuitable because optimal translations 
	require a more complete understanding of the structure of the sentence you're trying to reconstruct. 
	Consider the French: "Je n'ai pas un sandwich." Here, the 'pas' will only be processed after the 'avoir' instance, and so a greedy translation into English might give "I have not a sandwich".
**Q:** What numerical stability optimisation should you make?
	**A:** Maximise the $\log$ sum instead of the product, since otherwise the product will vanish due to underflow/rounding errors.
**Q:** What vulnerability is this algorithm susceptible to? How do you fix it?
	**A:** Longer sequences are comprised of more tokens, meaning the probability product is longer, and so the probability of those sequences arising is measured as smaller. Therefore, you want to perform _length normalisation_, where you divide by a $T_{y}^\alpha$ term, with $\alpha \approx 0.7$ being a good empirical approximation.

#### Error Analysis

Suppose that you have a human sequence $y^\ast$ and a sequence output by an RNN and beam search $\hat{y}$. To perform error analysis, you want to determine the likelihood of each sequence being output from the RNN. You then get two cases:

**Case 1:** $\Pr(y^*|x) > \Pr(\hat{y}|x)$

**Q:** Which part of the system is at fault? Why? 
	**A:** Here, Beam Search is at fault, since even though $y^\ast$ had a higher probability of emerging from the RNN, Beam Search selected $\hat{y}$ anyways, so you need to tune beam search.

**Case 2:** $\Pr(y^*|x) \leq \Pr(\hat{y}|x)$

**Q:** Which part of the system is at fault? Why? 
	**A:** Here, the RNN is at fault, since it wasn't trained suitably as to give $y^\ast$ the highest likelihood of occurring.

You can then repeat this for multiple output sets, to work out which fraction of errors arise from Beam Search/the decoder. Using this, you can then choose which one needs more tuning. Do you need to increase the beam width, or do further analysis on how to improve the RNN?


#### Bleu Score 

**Goal:** Automatically evaluate quality of machine translation

**Q:** Why does precision of translation in terms of whether or not each token appears in the references fail?
	**A:** Consider $x =$ "Le chat est sur le table". If we have $\hat{y} =$ "The the the the the the", then each 'the' appears in $x$, meaning it would have 100% precision, even though it's clearly a nonsense translation.

The _modified precision metric_, $P$, measures:

$$P = \frac{\sum_{i \in \hat{y}} \min \{ \;\text{Count}(i, \hat{y}), \text{Count}(i, y^\ast) \;\}  }{\sum_{i \in \hat{y}}\text{Count}(i, y^\ast)}$$

In other words, for each token $i$ in the output string, determine whether that string exists in a reference translation $y^\ast$, and only count its occurrences up to the number of times it appears in $y^\ast$. 

**NB:** There could be multiple reference translations for a given input, so we select the one which has the highest count of $i$, for each token.

We can then generalise this to work for any set of $n$-grams - that is, instead of having $i$ represent a token, have it represent unique $n$-grams in $\hat{y}$, and denote this score as $P_{n}$. 
To combine these scores together for multiple sets of $n$-grams, we can exponentiate their average:

$$\exp\left( \frac{1}{N} \sum_{n=1}^N P_{n} \right)$$

We also include a _brevity penalty_ coefficient: 

$$
K = 
\begin{cases} 
1 & \operatorname{len}(\hat{y}) > \operatorname{len}(y^\ast) \\
\exp\left( 1 - \frac{y^\ast}{\hat{y}} \right) & \text{otherwise}
\end{cases} 
$$

**Q:** Why do we use a brevity penalty? What does this represent?
	**A:** We want to punish sentences which are too long and too short, in order to give a faithful match to our intended translation.

Therefore, our final Bleu Score formula is:

$$B = K \exp\left( \frac{1}{N} \sum_{n=1}^N P_{n} \right)$$

### Attention Model

**Goal:** Shift away from the encoder-decoder model and give neural networks the ability to work with long sentences in fragments at a time, as a human translator might.
Original Paper: [[Bahdanau et al - Neural Machine Translation.pdf]]

**NB:** I've slightly butchered the explanation, and I think the paper does a really good job, so use that!

Use a BRNN (or any such flavour) to process input sequence and generate activations. Compute attention weights $\alpha^\time{x,y}$ for each output position $x$ from each input position $y$. Use those to generate context, which feeds into the RNN generating the output sequences. 

**Architecture:** (Tikz Diagram Coming Soon)
- Feed input sequence $x$ into a BRNN, $B$, (or some similar flavour, the important part is the bidirectional part)
- For each timestep cell in $B$, corresponding to position $t_x$ in $x$, concatenate the forward and backwards activations to give $a^\time{t_{x}}$, the activations for that cell. 
- Also have an RNN, $R$, which generates the output sequence, $y$. 
- For each timestep $t_y$ in generating $y$, feed _all_ the $a^\time{t_{x}}$s into a mini network ending in a softmax layer, whose result is $e^\time{t_{y}}$. Each entry $t_x$ of $e^\time{t_{y}}$ is the corresponding attention weight for the activation from input $x$, $\alpha^\time{t_{x}, t_{y}}$. 
- Again at each $t_y$, subsequently generate a _context vector_, $c^\time{t_{y}}$ from $e^\time{t_{y}}$ and the $a^\time{t_{x}}$s. 
- Feed $c^\time{t_{y}}$ and $y^\time{t_{y}-1}$ into $R$ to generate the new output, and iterate on $t_y$.

**Q:** Provide and explain the formula for $c^\time{t_{y}}$. 
	$$c^\time{t_{y}} = \sum_{t_{x}} \alpha^\time{t_{x}, t_{y}} a^\time{t_{x}}$$
	In other words, you're weighting the different input timesteps ($t_x$) for each new output timestep ($t_y$), to provide a suitable context at each stage, thereby allowing the windowing effect that a human translator would perform.

### Applications: Speech Recognition

We can apply the attention model to a sequence model, constraining the input BRNN and output RNN to have the same length. Where we can't identify the phoneme, or nothing is happening, we can include an empty character. 

**Connectionist Temporal Classification:** At each sample, identify the phoneme produced, and then when given that, compress all successive copies of the same phoneme not separated by the empty character.

**Q:** Why do we need CTC? 
	**A:** Audio is sampled such that there are necessarily more audio samples than phonemes in that track. We don't know how many phonemes a given audio track will contain (people have different speech patterns, among many other reasons), so we have to introduce the constraint that both RNNs have the same length, and then use CTC's compression rule.

**Q:** Give an example of how we can apply this to trigger word detection.
	**A:** We can extract audio features as a sequence from a given audio clip, and then add labels to each of those, indicating if the trigger word has just been said or not, and use this as the training set.

**NB:** Trigger Word Detection doesn't have a consensus on the 'best' algorithms yet.
# Week 4 - Transformers