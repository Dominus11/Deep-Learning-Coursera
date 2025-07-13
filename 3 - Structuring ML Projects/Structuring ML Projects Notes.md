### Introduction to ML Strategy

We don't want to waste loads of time for marginal improvements on model performance. There are so many ways you could seek to improve your models, and it's so easy to accidentally pick a direction that is fruitless. You need carefully deliberated strategy. 

**Orthogonalisation:** The process of separating tasks into independent/mutually exclusive processes. In the context of deep learning, we are referring to the notion of changing hyperparameters in a specific way to induce only one deliberate effect on the model. 

In ML, we aim to prove chain of assumptions, which is that:
- Fitting the training set well on the cost function
- $\implies$ The dev set is well fit to the cost function
- $\implies$ The test set is well fit on the cost function
- $\implies$ The model performs well in the real world

You want to orthogonalise the things we can do to effect each of these things, with such a decomposition from the previous courses being:
- Train Set: Bigger Network, Better optimiser like Adam. 
- Dev Set: Regularisation, Bigger Training Set
- Test Set: Bigger dev set, so that it captures more general ideas
- Real-World: Change either of the dev set or the cost function to capture the real-world performance better. 

### Goal Setting

#### Metrics

You ideally want a **single number evaluation metric** which can easily tell you about the performance of your model, so you can work out how to respond to the changes. The need for a _single_ metric is because multiple metrics can shift in opposite directions, making it more difficult to instantly tell which model you'd like to opt for when comparing 2 models. If you have multiple metrics, try to average them to yield your new single metric.

When it feels artificial to average your metrics, you might want to use the following:
- **Optimising Metrics:** Do as well as possible on this metric
- **Satisficing Metrics:** Just need to reach a minimum threshold of performance 

For $N$ metrics, it is common practice to have $1$ optimising metric, and $N-1$ satisficing metrics. 

#### Datasets

The reason your _dev and test sets need to come from the same distribution_ is that if they didn't, then you would be fitting your model to something that isn't representative of what it needs to be able to do. It's like doing past paper questions in Calculus for a paper on Linear Algebra. Your model won't be learning the necessary patterns to truly grasp the most general and useful mapping. 

Even then there's another layer though. Your model may not be a toy project on which your company is wasting hours, or even days and weeks, of GPU time! It needs to be able to perform in the real world. But it can only do that if it's learned the patterns that will be seen in the real world, so _pick your test set_ (and backpropagating through the previous point, your dev set) _to match what you will expect to see in practice_, and to _what you expect to be important_ to succeed on. 

Your test set is to measure confidence in the overall performance of the system. It needs to be big enough to give that confidence, and not much bigger. 

For some applications, you might not need that much confidence, so not having a test set is okay. Old machine learning practices used to tune to the test set (i.e. they had no test set, they had a dev set instead). If you're going to ship the final system regardless of its performance, which is not recommended, then you can omit the test set. 

#### Realignment 

Your metrics might not be perfect from the get-go! They may not suitably capture certain criteria you need to meet, and it's these kinds of issues that will motivate re-aligning your metrics. Such a way of doing this might include upgrading your average to be weighted against the true values of the training examples you're evaluating.  

This is an example of orthogonalisation as well, because you're separating out the tasks of:
1. Defining the metric, analogous to placing a target down
2. Performing well on the metric, analogous to learning how to aim the bow and arrow to hit the target. 

The same idea applies to your dev/test sets. Overall, if you have high performance on your metric and dev/test sets which doesn't then correspond to real world performance, you need to realign.

### Performance

The whole reason we're using deep learning is to try and establish a tool which can exceed human performance. What we observe is the beneath graph:

```latex
\usepackage{pgfplots}



```

We note that the gradient is relatively high up until human level competency is reached, and then it starts to flatten, asymptotically tending to a competency threshold which is known as the **Bayes Optimal Error**. Reaching this indicates that you have learned the best possible input to output mapping, which is made difficult due to particularly challenging input data (e.g. blurry images, fuzzy audio). 

While your model is worse than humans, you can improve by:
- Getting more annotated data
- Gaining insight from the analysis of manual error, since the model could be making the same mistakes, and has even learned these from the training data. 
- Bias-Variance analysis. Particularly you want to consider the margin of **avoidable bias** by measuring the difference between your Training Error and Bayes Optimal Error for the task. You shouldn't be exceeding Bayes' Optimal Error, because this will indicate that you're overfitting to the training data. A model trained on human-annotated data shouldn't be able to outperform the human annotation. 

We sometimes have to use human-level error as a proxy for the Bayes' error though, which makes sense because we can't define the upper limit on performance without deep analysis or simply being there already. We can at least bound it against the error peak measured human performance. 

When it comes to surpassing human performance though, it becomes a lot harder to quantify what Bayes' Error is, because you don't necessarily know the upper limit on performance anymore. It also becomes harder to intuitively work out what you can do to continue improving performance, since you'll seemingly have exhausted all human intuition on what could be fixed. 

It's an interesting observation to make that ML generally surpasses us on tasks involving large volumes of structured data, which aren't based on natural perception, since any model can view and learn the trends of more data than any one human could've. There are also some instances of speech/image recognition and medical tasks where ML has won out, but it was certainly harder to do these tasks. 

Supervised learning fundamentally assumes (in the spirit of orthogonalisation) that:
- You can fit the training set well, and can minimise the avoidable bias. To ensure this, train a bigger model, for longer, with better optimisation algorithms, search hyperparameter space thoroughly and consider the architecture. 
- Fitting the training set well implies that the model generalises well to the dev/test set, minimising the variance. For this, get more data, regularise, modify the hyperparameters. 


### Error Analysis

We are motivated once again by making the most productive efforts. So we'd like to have an error analysis framework that enables us to strategically distribute our efforts to the most pressing matters. 

To begin, you have to look at the model's output. It's the only way to get any sense of what the model hasn't been able to learn. 

It is a good idea to _estimate the ceiling on performance_, the maximum performance you could get from fixing an error in your dataset. You can then choose whether to proceed or not depending on whether you find the performance gain to be worth the work you'd have to do in getting it to work. You can also _try to evaluate multiple ideas in parallel_, by tabulating a sample of errors and seeing which ones are present, along with any comments, and see which task would enable the best improvement on performance, enabling you determine what the best options to pursue are. 

If you've got incorrectly labelled training examples that are reasonably random, you don't need to relabel the data. Especially if you have vast swathes of data. _DL algorithms are really quite robust to random errors_, which sort of makes sense. Systematic errors are problematic though, because, intuitively, this impacts the mappings learned. If you're very worried, you can address this in error analysis though, and address it if it makes up a significant proportion in your dev set error, since this affects your ability to select between two suggested models. 

Some things to consider when correcting dev and test set examples:
- _Whatever rectifying processes you apply to your dev set, apply to your test set too_, to maintain the distribution between the two. You don't desperately need to apply this to the significantly larger training set though, since it would be so much more time-consuming.  
- You also _might want to consider the examples the model got right, as well as those it got wrong_, since it might have seemingly got stuff right by dumb luck. People tend to avoid this because it's hard, and if your model is super accurate, it takes forever to check the stuff that was correct.

**Guideline:** Build your first system quickly, then iterate. This applies more strongly if you have less knowledge about your problem domain.

### Mismatched Dataset Distributions

Suppose you have very limited data for your domain-specific product, but you have some images from a slightly different, but close enough domain. Then you have two real options:
- Option 1: _Shuffle all the data in evenly_ and then split your train/dev/test as you would normally. On average, you aren't going to get a high proportion of the domain-specific data in your dev and test sets, meaning you won't be fitting very well to it. 
- Option 2: _Allocate all the domain-specific data to the dev/test set_ (mixed with a smaller proportion of the alternate domain data), and make the alternate domain data the training set. Then you can tune to the dev and test sets! This will give you better performance in the long run, since a higher proportion of what you're tuning/testing for will be relevant to your product. 

Suppose we use option 2, then we need to account for this in our Bias-Variance analysis. We no longer have orthogonalisation going from the training set to the dev set, since the distribution changed, and the data is fundamentally different in domain/origin/nature to that which it was trained on. Our solution is as follows:
- **Training - dev set:** A subset of the allocated training data that you now use for an additional form of dev testing, breaking down the change from training to dev into two steps, since this data now comes from the same domain as that it was trained on.
- The difference between training and the training-dev error will allow you to diagnose a variance problem now. To generalise this, hold the distribution/source of the data making your train set constant.
- The difference between the training-dev error and the dev error allows us to diagnose what we'll refer to as a **data mismatch problem**. You can generalise this by holding the experiment you're taking constant, be it human level or error on (non-)learned examples, and comparing between two different sources of data. 

So how do we address a data mismatch problem? There aren't many systematic ways.
- Carry out manual error analysis in order to work out the differences between the two datasets
- Make the training data more similar, or collect more data which is similar to the dev/test sets. 

To the effect of achieving the second one, an interesting approach will be to use _artificial data synthesis_, but you don't want to run into the _problem of synthesising a small subset of the possible data domain_, since you could end up overfitting your model to that. Even if the synthesised data may be indistinguishable to a human, your model will be able to see the differences between the rest of the domain, which results in this overfitting issue.

### Learning from Multiple Tasks

#### Transfer Learning

**Transfer Learning:** The idea of taking knowledge learned by a neural network for one task $A$ to a separate task $B$, under the following criteria:
- $A$ and $B$ share the same input
- There is more data for $A$ than $B$ 
- Low-level features from $A$ could be helpful for learning $B$

This is actually very simple to implement in practice:
- Train the for the first time on data for task $A$. 
- Re-initialise the weights randomly for the last (few) layer(s). Alternatively, you can add on some more layers in their place. 
- Feed in the new data from task $B$ and retrain only these last few layers. 

With sufficient data, you could retrain the entirety of the neural network, described in terms of:
- _Pre-Training:_ Training for the first time for task $A$ to learn a set of initial parameters. 
- _Fine-Tuning:_ Using the learned parameters from the pre-training as your initialisations, retrain the model on the data for task $B$. 

#### Multi-task Learning

**Multi-task Learning:** Trying to learn multiple tasks in parallel, with each task hopefully helping all of the other tasks. 

There isn't anything fundamentally different to the regular process of training a neural network! But it is more difficult due to the caveats required to make it suitable, and the training times for a more complex neural network.

This is appropriate when:
- You're training a set of tasks that could benefit from having shared lower-level features
- While not a hard-and-fast rule, the amount of data for each task is similar. The principle is that the other tasks have many more examples than what you have for this one task.
- You can train a big enough neural network to do well on all the tasks. 

In practice, Transfer Learning is used a lot more than Multi-task Learning, with the main exception being the field of Computer Vision. 
### End-to-end Deep Learning

**End-to-end Deep Learning:** This replaces a multi-stage machine learning/data pipeline with one large neural network that can perform the entire task. 

This approach only really shines with huge amounts of data, on the order of 10,000 - 100000h. There are intermediate versions depending on the scale of data available, which vary the scale of decomposition, like these:

One example of decomposing the data pipeline is _face recognition_ for access! The best approach is to:
- Detect the person's face, and zoom in on this and crop it so that the person's face is centred, using one neural network. 
- Verify if this is the person using another neural network to determine if this is a person in your employee database. 

The reason it's better to decompose is because there is a lot of data for each sub-task. There's lots of facial data to learn the general idea of identifying someone in a huge facial database, and then you could apply transfer learning. There's also lots of data to practice zooming in and centring on a face. 

Other examples of decomposing the pipeline include:
- Machine Translation systems, by first doing some text analysis before performing the translation with an NN.
- Estimating a child's age given medical images, for paediatricians to determine if a child is aging properly. 

End-to-End deep learning:
- Lets the data speak for itself, since the network simply learns the direct mapping 
- Uses less hand-designed components, but this is a double edged sword, since you could inject some very useful knowledge into the network, or you could be preventing it from learning the most efficient/elegant representation. 
- Requires a _vast amount of data, which is the main limiting variable_ in asking whether you can reasonably use it for a task. You need lots of data in order to capture the complexity of the mapping between $X$ and $Y$. 