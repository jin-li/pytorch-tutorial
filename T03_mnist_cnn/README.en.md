语言：[简体中文 🇨🇳](README.md) | Language: English 🇺🇸

# PyTorch MNIST Handwritten Digit Recognition (3)

This folder contains the code used in the article [Learn PyTorch by Examples (3): Implementing MNIST Handwritten Digit Recognition with Convolutional Neural Networks (III)](https://jinli.io/en/p/learn-pytorch-by-examples-3-implementing-mnist-handwritten-digit-recognition-with-convolutional-neural-networks-iii/).

## Quick Start

First, you need to create a Python virtual environment to run this project. You can use `virtualenv` or `conda` and other tools to create a virtual environment. You can refer to my article [Python Environment Management with venv/conda/mamba](https://jinli.io/en/p/python-environment-management-with-venv/conda/mamba/).

1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Run the code:

    ```bash
    python plot_performance.py
    ```

3. Recognize handwritten digits:

    You can create a handwritten digit image to recognize, and then run `classify.py` to recognize this image. I provide some example images that I wrote in the `numbers` folder, you can use these images to test.

    ```bash
    python classify.py numbers/number0.png
    ```