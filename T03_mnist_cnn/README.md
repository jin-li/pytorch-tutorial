语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# PyTorch MNIST 手写数字识别（三）

本文件夹包含了[《实例学PyTorch（3）：使用卷积神经网络实现MNIST手写数字识别（三）》](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch3%E4%BD%BF%E7%94%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%AE%9E%E7%8E%B0mnist%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%B8%89/)文章中使用的代码。

## 快速开始

首先你需要创建一个运行本项目的Python虚拟环境，你可以使用`virtualenv`或者`conda`等工具来创建虚拟环境。可参考我的文章[《Python环境管理方式总结》](https://jinli.io/p/python%E7%8E%AF%E5%A2%83%E7%AE%A1%E7%90%86%E6%96%B9%E5%BC%8F%E6%80%BB%E7%BB%93/)。

1. 安装依赖：

    ```bash
    pip install -r requirements.txt
    ```

2. 运行代码：

    ```bash
    python plot_performance.py
    ```

3. 识别手写数字：

    你可以自己创建一个要识别的手写数字图片，然后运行`classify.py`来识别这个图片。我在`numbers`文件夹中提供了一些我自己写的示例图片，你可以使用这些图片来测试。

    ```bash
    python classify.py numbers/number0.png
    ```