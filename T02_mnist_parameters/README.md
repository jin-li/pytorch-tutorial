语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# PyTorch MNIST 手写数字识别（二）

本文件夹包含了[《实例学PyTorch（2）：MNIST手写数字识别（二）——神经网络中的参数选择》](https://blog.jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch2mnist%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%BA%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E5%8F%82%E6%95%B0%E9%80%89%E6%8B%A9/)文章中使用的代码。

## 快速开始

首先你需要创建一个运行本项目的Python虚拟环境，你可以使用`virtualenv`或者`conda`等工具来创建虚拟环境。可参考我的文章[《Python环境管理方式总结》](https://blog.jinli.io/p/python%E7%8E%AF%E5%A2%83%E7%AE%A1%E7%90%86%E6%96%B9%E5%BC%8F%E6%80%BB%E7%BB%93/)。

1. **安装依赖**，这里提供两种方式：

    - `conda`/`mamba`/`micromamba`/`miniconda`等（**推荐**）：

        ```bash
        conda env create -f environment.yml
        ```

        若使用`mamba`或`micromamba`，请将上面的命令中的`conda`替换为`mamba`或`micromamba`。
    
    - `pip`：

        若使用`pip`，请确保python版本为3.11。我试过Python 3.12，安装`torchvision`时会报版本不匹配的错误。我没有测试过更低版本的Python。

        ```bash
        pip install -r requirements.txt
        ```

2. **运行代码**：

    ```bash
    python parametric_study.py
    ```