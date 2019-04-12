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