# Short summary of some things in the course

# Entropy

### Shannon Entropy
It is defined as,

<img src="https://render.githubusercontent.com/render/math?math=H = - \sum_{i} p_{i} log_{b}p_{i}">

Example:
Consider a single toss of skewed coin (it is likely to show one side more than the other side). Regarding the uncertainty of the outcome {head, tail}.

A fair coin that lands 50\% of the time each on heads and tails produces exactly one bit (0 or 1) of entropy per toss. However, if you had a coin that wasn't fair, it produces a fraction of a bit of entropy.
Thus the correct answer is that,

**The entropy is smaller than one bit**

# Decision Trees
To be added.


# Challenges in ML

#### The curse of dimensionality

#### The bias-variance tradeoff

# Regression

### RANSAC - RANdom SAmpling Consensus

*Repeat M times:*
1. Select a random subset of the original data
2. Fit model to the subset
3. Test all other data against the model. Points which fit the model well according to some loss function are considered to be part of the consensus set
4. The model is reasonably good if sufficiently many points have been classified as part of the consensus set

<img src="media/ransac.gif" alt="drawing" width="400"/>

### Nearest Neighbor Regression


<!-- # Probabilistic Reasoning
To be added. -->

<!-- # Learning as Inference -->
<!-- To be added. -->

# Priors and Latent Variables
### K-means

K-means is an unsupervised algorithm which is used to identify clusters in data.

1. Choose number of clusters *K*
2. Initialize centroids by first shuffling the data set and then randomly select without replacement *K* data points to use as centroids
3. Compute the distances between the data points and all centroids
4. Assign each data point to the closest centroid
5. Update centroids by taking the average of all data points that belong to each cluster

Repeat steps 3-5


<img src="media/kmeans.gif" alt="drawing" width="250">

### Expectation maximization
Expectation maximization(EM) is an iterative method to find *maximum likelihood* (ML) or *maximum a posteriori* (MAP) estimates of parameters in statistical models, where the model depends on unobserved latent variables.

<img src="media/em.gif" alt="drawing" width="250">

# Artificial Neural Networks

### Perceptron
A perceptron is a simple mathematical model of a neuron.
A single perceptron can serve as a classifier or regressor. It is a building block in a more complex artifical neural network structures which are used to solve advanced problems.

The perceptron consists of
1. An input layer
2. Weights and Bias used to form a net sum
3. A threshold in the form of an Activation Function. (Maps the input to desired values such as {0,1})
4. An output layer

<img src="media/perceptron.png" alt="drawing" width="400">

The network is trained by adjusting the weights and bias each node. Adjustments are made based on the error produced.

A perceptron is a single layer neural network and a multi-layer perceptron is called a Neural Network.

### Backpropagation
Short for backward propagation of errors, is an algorithm for supervised learning of artificial neural networks using gradient descent.

<img src="media/neuralnet.gif" alt="drawing" width="300">

Since we cannot simply train neurons in a network as the desired output is unknown for neurons in the middle layers, we'll use Backpropagation.
Given an error function, the method calculates the gradient of the error function with respect to the neural network's weights. The calculation then proceeds backwards through the network

> Gradient descent is an iterative optimization algorithm for finding the minimum of a function; in our case we want to minimize the error function. To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.

There are three steps in this method:


#### Forward step
*Output values for the nodes*

>We'll use the initial weights and inputs to predict the output. The inputs are multiplied by weights and then passed forward to the next layer.


#### Backward propagating step
*Compute local (generalized) errors*
>The generalized errors are first computed at the output nodes, where the targets are known, and then propagated backwards to assign local errors to every node in the network

#### Local step
*Update weights*
> Once the output values and generalized errors are known for each node in the network, the weights can be updated. This can be done locally since no more global communication is needed.


### Dropout

Large neural nets trained on relatively small datasets can overfit the training data.
Dropout is regularization method for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is a very efficient way of performing model averaging with neural networks.


> During training, some number of layer outputs are randomly ignored or "dropped out". This has the effect of making the layer look like and be treated like a layer with a different number of nodes and connectivity to the prior layer. In effect, each update to a layer during training is performed with a different "view" of the configured layer.

<img src="media/dropout.gif" alt="drawing" width="300">

# Support Vector Machines
To be added

# Ensemble Methods

### Bagging

### Decision Forests

# Dimensionality Reduction

### Principal Component Analysis

### Concept of subspace and subspace methods

### Similarity Measures

### Fisher's Criterion