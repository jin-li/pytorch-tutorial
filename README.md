语言：简体中文 🇨🇳 | Language: [English 🇺🇸](README.en.md)

# 实例学PyTorch

## 简介

本项目是《实例学PyTorch》系列文章的配套代码，包括了文章中的所有代码实例。

本项目基于PyTorch官方发布的示例，对示例代码进行了适当地改造和详细地解释，添加了必要的注释，并加入了很多的工具性的代码以帮助分析对比。

作者在自学PyTorch的过程中，发现PyTorch官方的示例代码虽然很不错，但对于初学者来说可能并不容易上手。因此作者根据自己的学习经验，对官方示例代码进行了适当地改造和详细地解释，来帮助初学者更好地理解机器学习和深度学习的基础知识，以及这些知识是如何通过PyTorch应用到实际的代码中的。

## 文章列表

《实例学PyTorch》系列文章发布在作者的个人博客上，文章有中文版和英文版，文章列表如下：

- [实例学PyTorch（1）：MNIST手写数字识别（一）——PyTorch基础和神经网络基础](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch1mnist%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%B8%80pytorch%E5%9F%BA%E7%A1%80%E5%92%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80/)
- [实例学PyTorch（2）：MNIST手写数字识别（二）——神经网络中的参数选择](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch2mnist%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%BA%8C%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E4%B8%AD%E7%9A%84%E5%8F%82%E6%95%B0%E9%80%89%E6%8B%A9/)
- [实例学PyTorch（3）：MNIST手写数字识别（三）——卷积神经网络（CNN）](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch3mnist%E6%89%8B%E5%86%99%E6%95%B0%E5%AD%97%E8%AF%86%E5%88%AB%E4%B8%89%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9Ccnn/)
- [实例学PyTorch（4）：序列预测（一）——循环神经网络（RNN）](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch4%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E4%B8%80%E5%BE%AA%E7%8E%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9Crnn/)
- [实例学PyTorch（5）：序列预测（二）——门控循环单元（GRU）和长短期记忆网络（LSTM）](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch5%E5%BA%8F%E5%88%97%E9%A2%84%E6%B5%8B%E4%BA%8C%E9%97%A8%E6%8E%A7%E5%BE%AA%E7%8E%AF%E5%8D%95%E5%85%83gru%E5%92%8C%E9%95%BF%E7%9F%AD%E6%9C%9F%E8%AE%B0%E5%BF%86%E7%BD%91%E7%BB%9Clstm/)
- [实例学PyTorch（6）：语言模型（一）——使用LSTM实现词级语言模型](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch6%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%B8%80%E4%BD%BF%E7%94%A8lstm%E5%AE%9E%E7%8E%B0%E8%AF%8D%E7%BA%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/)
- [实例学PyTorch（7）：语言模型（二）——使用Transformer实现词级语言模型](https://jinli.io/p/%E5%AE%9E%E4%BE%8B%E5%AD%A6pytorch7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B%E4%BA%8C%E4%BD%BF%E7%94%A8transformer%E5%AE%9E%E7%8E%B0%E8%AF%8D%E7%BA%A7%E8%AF%AD%E8%A8%80%E6%A8%A1%E5%9E%8B/)

## 快速开始

各个项目的代码分别放在各自的文件夹中，文件夹的编号和文章的编号对应。每个文件夹中都有一个`README.md`文件，包含了运行代码的步骤。

- 读者若只想运行某个项目，可阅读该项目的`README.md`文件，按照其中的步骤进行操作。
- 读者若想详细学习了解某个项目，可阅读该项目对应的文章。

各个项目所需的Python环境是分离的，每个项目所需的依赖在文件夹中的`requirements.txt`文件中列出。读者可以根据需要创建虚拟环境，然后安装依赖。关于Python虚拟环境的创建和管理，可参考作者的文章[《Python环境管理方式总结》](https://jinli.io/p/python%E7%8E%AF%E5%A2%83%E7%AE%A1%E7%90%86%E6%96%B9%E5%BC%8F%E6%80%BB%E7%BB%93/)。

## 鸣谢

本文中部分示例代码参考或改编了PyTorch官方发布的[示例代码](https://github.com/pytorch/examples).