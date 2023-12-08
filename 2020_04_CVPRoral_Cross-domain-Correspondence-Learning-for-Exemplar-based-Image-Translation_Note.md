# Cross-domain Correspondence Learning for Exemplar-based Image Translation

> "Cross-domain Correspondence Learning for Exemplar-based Image Translation" CVPRoral, 2020 Apr, `CoCosNet`
> [paper](http://arxiv.org/abs/2004.05571v1) [code](https://github.com/microsoft/CoCosNet) [website](https://panzhang0212.github.io/CoCosNet/) [blog](https://zhuanlan.zhihu.com/p/336722743)
> [pdf](./2020_04_CVPRoral_Cross-domain-Correspondence-Learning-for-Exemplar-based-Image-Translation.pdf)
> Authors: Pan Zhang, Bo Zhang, Dong Chen, Lu Yuan, Fang Wen

## Key-point

- Task
- Problems
- :label: Label:

可以参考一些 Exemplar-based image synthesis 方法，如何做 alignment



## Contributions



## Introduction

看 website 里面列出的应用：image translation, makeup transfer, image editing
![](https://panzhang0212.github.io/CoCosNet/images/makeup.gif)



### SPADE

> "Semantic Image Synthesis with Spatially-Adaptive Normalization"
> [paper](https://arxiv.org/abs/1903.07291) [code](https://github.com/NVlabs/SPADE) [blog](https://zhuanlan.zhihu.com/p/373068906)

本文的模型架构基于 SPADE 来做；

模型整体结构

![](https://pic1.zhimg.com/80/v2-d0b096d36d4381b9340ac4c77df446e8_720w.webp)



**SPADE block 结构**

![](https://pic4.zhimg.com/80/v2-bffc4768d4e811d1f7ae86ee95458b1b_720w.webp)

在每层应用SPADE之前先resize到相应layer的feature map大小，然后再卷积分别生成 bete 和 gamma

![](https://pic3.zhimg.com/80/v2-978b2f5fc54cf34664fe5383a9efc2a6_720w.webp)



输入 exemplar 图像提取特征，学习后续的 $\mu, \sigma^2$ 参数

![](https://pic2.zhimg.com/80/v2-a241e279a4db9ca1ff04dabfee8a9729_720w.webp)





## methods

> - FPN?

网络包含两个部分：跨域对齐网络和图像生成网络

使用的映射方式是用FPN提取两张图像的特征图，再都转换为S域中的表示XS和YS，这里的F就是整个的转换关系，theta是需要学习的参数



### Translation network

>  "Semantic Image Synthesis with Spatially-Adaptive Normalization"
> [paper](https://arxiv.org/abs/1903.07291)

设计参考 "Semantic Image Synthesis with Spatially-Adaptive Normalization"

![](https://pic4.zhimg.com/80/v2-955ac7ad7a749a0994540d5ea831acbf_720w.webp)

> 从 warped exemplar 里面学习参数 $\theta$ 输入到生成网络里面

接下来是它的图像生成网络，从一个固定的常量z开始，通过逐步卷积逐步注入扭曲图像的风格信息，每一次注入风格都是通过Positional normalization和Spatially-adaptive denormalization, positional normalization是指在每一个像素点进行归一化操作，SPADE指的是去正则化时的Alpha和Beta不是学来的，而是从风格参考图像中得来的，而且也是每一个像素不同，这里得到Alpha和Beta的操作也是通过卷积。



### Loss

loss 在此任务中很关键，约束 condition 的 encoder 学习的特征





## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

