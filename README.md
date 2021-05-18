# Avans_NeuralNetworking
This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 255.

In this example, each image will be converted to float32, normalized to [0, 1] and flattened to a 1-D array of 784 features (28*28).

# System requirements 
Python 3.6–3.8
Python 3.8 support requires TensorFlow 2.2 or later.
pip 19.0 or later (requires manylinux2010 support)
Ubuntu 16.04 or later (64-bit)
macOS 10.12.6 (Sierra) or later (64-bit) (no GPU support)
macOS requires pip 20.3 or later
Windows 7 or later (64-bit)
Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019
GPU support requires a CUDA®-enabled card (Ubuntu and Windows)

## Installation
1. Install the Python development environment on your system 
Check if your Python environment is already configured:

Requires Python 3.6–3.8, pip and venv >= 19.0

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install Tensorflow.

```bash
python3 --version
pip3 --version
```

2. Install Tensorflow with the PIP Command line
```
pip3 install --user --upgrade tensorflow  # install in $HOME
```

## Usage
1. Setup the input parameters of the Neural network to increase/decrease performance.
2. Choose the dataset that must be executed
3. Run the main.py file
