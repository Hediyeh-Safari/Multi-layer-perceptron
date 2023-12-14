# Multi-layer-perceptron
In this task, you will code the backpropagation algorithm to train a multi-layer perceptron for classifying data into one of four distinct classes. The implementation must be done from scratch, without relying on established libraries such as Tensorflow, Keras, or PyTorch.

Two files, "train data.csv" and "train labels.csv," are provided, containing a dataset of 24,754 samples. Each sample has 784 features, categorized into four classes (0, 1, 2, 3). Your task includes splitting the data into training and validation sets to prevent overfitting. Finally, your model will be evaluated on an unseen test set.

The architecture of your neural network should consist of one input layer, one hidden layer, and one output layer. The labels are in one-hot encoded format, where class 0 is represented as [1, 0, 0, 0], and class 2 is represented as [0, 0, 1, 0]. Ensure the appropriate activation function is used in the output layer. The number of nodes in the hidden layer can be chosen as needed.

You are required to create a single function that enables the utilization of the trained network for predicting the test set. This function should output the labels in a one-hot encoded format, presented as a numpy array.
