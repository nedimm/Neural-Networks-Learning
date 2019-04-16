# Neural Networks Learning

In this project we will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. The project is an exercise from the ["Machine Learning" course](https://www.coursera.org/learn/machine-learning/) from Andrew Ng.

To get started with the project, you will need to download the starter code and unzip its contents to a directory. There are 5000 training examples in `ex4data1.mat`. Starting point of the project is the `ex4.m` Octave script.

## Neural Networks

In the previous [project](https://github.com/nedimm/Multi-class-Classification-with-Neural-Networks), we implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we
provided. In this project, we will implement the backpropagation algorithm to learn the parameters for the neural network.

### Visualizing the data

Let's first visualise the training data and display it on 2-dimensional plot by calling the function `displayData`: 

![vis](https://i.paste.pics/590kn.png)
***Figure 1: Example of a training data***

Each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix `X`. This gives us a 5000 by 400 matrix `X` where every row is a training example for a handwritten digit image.

![x](https://i.paste.pics/590PR.png)

The second part of the training set is a 5000-dimensional vector `y` that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index, the digit zero has been mapped to the value ten. Therefore, a `0` digit is labeled as `10`, while the digits `1` to `9` are labeled as `1` to `9` in their natural order.

### Model representation

The neural network is shown in `Figure 2`. It has 3 layers - and input layer, a hidden layer and an output layer. Recall that the inputs are pixel values of digit images. Since the images are of size 20 × 20, this gives us 400 input
layer units (not counting the extra bias unit which always outputs +1). The training data will be loaded into the variables `X` and `y` by the `ex4.m` script.
You have been provided with a set of network parameters $$(Θ^{(1)}, Θ^{(2)})$$ already trained. These are stored in `ex4weights.mat` and will be loaded by `ex4.m` into `Theta1` and `Theta2`. The parameters have dimensions
that are sized for a neural network with 25 units in the second layer and 10 output units (corresponding to the 10 digit classes).

![code](https://i.paste.pics/59Y3I.png)

![f2](https://i.paste.pics/59Y3S.png)
***Figure 2: Neural network model***

### Feedforward and cost function
Now we will implement the cost function and gradient for the neural network. First, we have to complete the code in `nnCostFunction.m` to return the cost.
Recall that the cost function for the neural network (without regularization) is:

![](https://i.paste.pics/59Y4T.png)

where $$h_θ(x^{(i)})$$ is computed as shown in the `Figure 2` and `K = 10` is the total number of possible labels. Note that $$h_θ(x^{(i)})_k = a^{(3)}_k$$ is the activation (output value) of the `k-th` output unit. Also, recall that whereas the original labels (in the variable y) were 1, 2, ..., 10, for the purpose of training a neural network, we need to recode the labels as vectors containing only values 0 or 1, so that

![](https://i.paste.pics/59Y5L.png)

For example, if $$x^{(i)}$$ is an image of the digit 5, then the corresponding $$y^{(i)}$$ (that we should use with the cost function) should be a 10-dimensional vector with $$y_5 = 1$$, and the other elements equal to 0. We should implement the feedforward computation that computes $$h_θ(x^{(i)})$$ for every example `i` and sum the cost over all examples. Our code should also work for a dataset of any size, with any number of labels (we can assume that there are always at least `K ≥ 3` labels).

![](https://i.paste.pics/59Y70.png)

Using the formulas from `Figure 2` and the formula for cost function (without regularization) we get:

```matlab
a1 = [ones(m, 1) X];

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1), 1) a2];

z3 = a2 * Theta2';
a3 = sigmoid(z3);
hThetaX = a3;

yVec = zeros(m,num_labels);

for i = 1:m
    yVec(i,y(i)) = 1;
end

J = 1/m * sum(sum(-1 * yVec .* log(hThetaX)-(1-yVec) .* log(1-hThetaX)));
```

After implementing this step, we can verify that our cost function computation is correct by verifying the cost computed in `ex4.m`:

![](https://i.paste.pics/5AO9W.png)

### Regularized cost function

The cost function for neural networks with regularization is given by

![](https://i.paste.pics/5AOK5.png)

After adding the regularization part to the calculation of the cost function we can verify that the result is correct by comparing it with the value in `ex4.m`

```matlab
regularator = (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))) * (lambda/(2*m));
J = J + regularator;
```

![](https://i.paste.pics/5AOMC.png)

## Backpropagation

In this part of the project, we will implement the backpropagation algorithm to compute the gradient for the neural network cost function. We will need to complete the `nnCostFunction.m` so that it returns an appropriate value for `grad`. Once we have computed the gradient, we will be able to train the neural network by minimizing the cost function `J(Θ)` using an advanced optimizer such as `fmincg`.
We will first implement the backpropagation algorithm to compute the gradients for the parameters for the (unregularized) neural network. After we have verified that your gradient computation for the unregularized case is correct, we will implement the gradient for the regularized neural network.

### Sigmoid gradient

We will first implement the sigmoid gradient function. The gradient for the sigmoid function can be
computed as 

![](https://i.paste.pics/5AOO4.png)

where 

![](https://i.paste.pics/5AOOO.png)

```matlab
g = sigmoid(z) .* (1 - sigmoid(z));
```

### Random initialization

When training neural networks, it is important to randomly initialize the parameters for symmetry breaking. One effective strategy for random initialization is to randomly select values for $$Θ^{(l)}$$ uniformly in the range $$[- \epsilon_{init}, \epsilon_{init}]$$. 
One effective strategy for choosing $$\epsilon_{init}$$ is to base it on the number of units in the network. A good choice of $$\epsilon_{init}$$ is $$\epsilon_{init} = \frac{\sqrt{6}}{\sqrt{L_{in}+L_{out}}}$$ , where $$L_{in}=s_{l}$$ and $$L_{out} = s_{l+1}$$ are the number of units in the layers adjacent to $$Θ^{(l)}$$.
We will use $$\epsilon_{init}=0.12$$. This range of values ensures that the parameters are kept small and makes the learning more efficient.
Our job now is to complete `randInitializeWeights.m` to initialize the weights for Θ; modify the file and fill in the following code:
```matlab
% Randomly initialize the weights to small values
epsilon_init = 0.12;
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;
```

### Backpropagation

Now, we will implement the backpropagation algorithm. Recall that the intuition behind the backpropagation algorithm is as follows. Given a training example $$(x^{(t)}, y^{(t)})$$, we will first run a "forward pass" to compute all the activations throughout the network, including the output value of the hypothesis $$h_Θ(x)$$. Then, for each node `j` in layer `l`, we would like to compute an "error term" $$\delta_j^{(l)}$$ that measures how much that node was "responsible" for any errors in our output.
For an output node, we can directly measure the difference between the network’s activation and the true target value, and use that to define $$\delta_j^{(3)}$$ δj (since layer 3 is the output layer). For the hidden units, we will compute $$\delta_j^{(l)}$$ based on a weighted average of the error terms of the nodes in layer (l + 1).

The backpropagation algorithm is depicted in Figure 3.

![](https://i.paste.pics/5B6IX.png)
***Figure 3 Backpropagation Updates***





