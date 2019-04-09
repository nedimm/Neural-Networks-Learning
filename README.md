# Neural Networks Learning

In this project we will implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition. The project is an exercise from the ["Machine Learning" course](https://www.coursera.org/learn/machine-learning/) from Andrew Ng.

To get started with the project, you will need to download the starter code and unzip its contents to a directory. There are 5000 training examples in `ex4data1.mat`. Starting point of the project is the `ex4.m` Octave script.

## Neural Networks

In the previous [project](https://github.com/nedimm/Multi-class-Classification-with-Neural-Networks), we implemented feedforward propagation for neural networks and used it to predict handwritten digits with the weights we
provided. In this project, we will implement the backpropagation algorithm to learn the parameters for the neural network.

### Visualizing the data

Let's first visualise the training data and display it on 2-dimensional plot by calling the function `displayData`: 

![vis](https://i.paste.pics/590kn.png)
***Figure 1: Example of a training data

Each training example is a 20 pixel by 20 pixel grayscale image of the digit. Each pixel is represented by a floating point number indicating the grayscale intensity at that location. The 20 by 20 grid of pixels is "unrolled" into a 400-dimensional vector. Each of these training examples becomes a single row in our data matrix `X`. This gives us a 5000 by 400 matrix `X` where every row is a training example for a handwritten digit image.

![x](https://i.paste.pics/590PR.png)

The second part of the training set is a 5000-dimensional vector `y` that contains labels for the training set. To make things more compatible with Octave/Matlab indexing, where there is no zero index, the digit zero has been mapped to the value ten. Therefore, a `0` digit is labeled as `10`, while the digits `1` to `9` are labeled as `1` to `9` in their natural order.
