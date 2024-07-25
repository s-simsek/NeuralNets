<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="images/neuralnets.jpg" alt="Logo" width="700" height="200">
  </a>
  <h3 align="center">Neural Networks From Scratch</h3>
  <p align="center">
    Full Implementation of Fully Connected Neural Networks with plain Numpy
  </p>
</div>

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/s-simsek/NeuralNets.git
    ```
2. Navigate to the project directory:
    ```sh
    cd NeuralNets
    ```
3. Create a virtual environment and download the required dependencies:
    ```sh
    virtualenv venv
    source venv/bin/activate 
    pip3 install -r requirements.txt
    ```
## Notes
`gradient_check.ipynb` verifies the backpropagation of loss and activation functions

`optimizer_check.ipynb` verifies the implementation of optimizers (SGD, RMSprop, RMSprop w/ Nesterov Momentum, Adam, Nadam)

`demo.ipynb` shows how to use the Neural Network module and compare the implementation with similar TensorFlow model

## Resources 

[Homework 6](https://people.eecs.berkeley.edu/~jrs/189s21/) from UC Berkeley's CS189 Intro to ML class

[Dive into Deep Learning](https://d2l.ai/)

[Deep Learning Book](https://www.deeplearningbook.org/)
