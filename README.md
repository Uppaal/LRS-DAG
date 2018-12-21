# LRS-DAG
Non Adversarial Domain Adaptation, in a Low Resource Supervised Setting

This repository consists of a method (presented [here](https://drive.google.com/file/d/1ceyg2lOjqed5l1IrWz8A5oFQEvEGwFf3/view?usp=sharing)) applied to two models: a basic Fully Connected Network and a standard CNN. Three datasets have been used: MNIST, SVHN and a synthetic dataset based off MNIST called Syn-MNIST.

- To run the basic fully connected model, run python fc.py
- Change the target dataset between MNIST, SVHN, Syn-MNIST using flags specified in the main function.
- To run the convolutional neural network, run python cnn.py
