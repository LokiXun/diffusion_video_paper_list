# GLEAN: Generative Latent Bank for Image Super-Resolution and Beyond

> "GLEAN: Generative Latent Bank for Image Super-Resolution and Beyond" TAPMI, 2022 Jul :star:
> [paper](https://arxiv.org/abs/2207.14812) [code](https://github.com/open-mmlab/mmagic?utm_source=catalyzex.com) [video_explanation](https://www.youtube.com/watch?v=73EqkLim41U) :+1:
> [paper local pdf](./2022_07_TPAMI_GLEAN--Generative-Latent-Bank-for-Image-Super-Resolution-and-Beyond.pdf)
>
> 使用 StyleGAN 大模型先验，从里面抽一些特征辅助进行 SR。参考同样方式做 Diffusion
>
> - :question: 高 quality 和高 fidelity 的图像定义和区别？

## **Key-point**

- Task: 用预训练 StyleGAN 先验做图像超分
- Background

提出 `encoder-bank-decoder` 框架，**利用预训练生成网络作为 Latent bank 提供先验信息，实现生成高 quality（结构信息） 和高 fidelity （结果是否自然）的图像**。直接用 ESRGAN 网络生成，得到的结果可信度和质量都不太行。例如：ESRGAN 生成的结构信息相似，但细节不自然；PULSE （GAN-inversion）方法生成更自然，但结构不相似；

**GLEAN 融合两个方法，有效在高质量 & 高可信度方面，提升生成图像质量：Encoder + Decoder 有效融合结构信息，保证结构相似 & 融合预训练 StyleGAN 先验，实现细节更自然**

**提出的 Latent Bank 全新框架， 验证了融合结构信息 & 自然图像的先验，很有必要 :star:**

![image-20230921154117103](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921154117103.png)

- 场景
  1. GLEAN 提供额外的自然图像先验，针对高倍率 SR 和输入图像退化很严重时候，能够有效输出
  2. 可以用到其他图像修复任务，denoise，colorization



## **Contributions**

- GLEAN 有效在高质量 & 高可信度方面，提升生成图像质量
  Encoder + Decoder 有效融合结构信息，保证结构相似 & 融合预训练 StyleGAN 先验，实现细节更自然
- 生成质量好 & 能用到高倍率超分 （x64）& 泛化性好（不同先验，对应到各个类别；能在真实图像）
- 提出的 Latent Bank 全新框架， 验证了融合结构信息 & 自然图像的先验，很有必要 
- 提取轻量化 Light GLEAN



## **Related Work**

SRCNN, SRGAN 通过 Encoder & Decoder & （pixel loss, adversarial loss）训练。Train from scratch，网络需要同时负责提取 natural image 特征 & 维持保真度，同时要满足这些要求很苛刻，**输出的图像在纹理和细节不太好**

- SRGAN
  [blog](https://medium.com/analytics-vidhya/super-resolution-gan-srgan-5e10438aec0c)

- ESRGAN

  [blog](https://medium.com/analytics-vidhya/esrgan-enhanced-super-resolution-gan-96a28821634) :+1:

  ![SRGAN](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*m92XZhAi6AKmnHur-4cNjw.png)

  提出用 Residual-in-Residual Dense Block (RRDB) 替换原来的 Residual Block

  1. Removal of all Batch Normalization (BN) layers
  2. Replacing the original basic block with the RRDB

  ![](https://miro.medium.com/v2/resize:fit:1400/format:webp/1*UhYRqN7-rkzi3cN4qYRdAA.png)



GAN inversion 利用低维 latent code 引导先验网络，保真度还可以，结构和原来差距很多（复原图像和原图很不相似）

发现需要**额外的结构信息**保持结构信息

- 因此 GLEAN 通过融合 Encoder+Decoder（维持结构信息），GAN-inversion 方式维持生成细节和可信度。融合结构信息 & 自然图像先验，实现高质量（结构相似）&高保真度（生成是否自然）



### PULSE

> "PULSE: Self-Supervised Photo Upsampling via Latent Space Exploration of Generative Models"
> [code](https://github.com/krantirk/Self-Supervised-photo) [blog](https://medium.com/@joniak/keep-your-finger-on-the-pulse-of-the-super-resolution-5201a855e1a0)

对 LR 图像 $I_{LR}$ 做超分，给定一堆 HR 图像，PULSE seeks for for a latent vector $z\in \cal{L}(latent~space)$ that minimizes  $downscaling~loss = \abs{\abs{DS(G(z)) - I_{LR}}}_p^p < \epsilon(1e{-3})$ ，$I_{SR}=G(z)$, $DS$ 代表下采样

- 缺点：推理很慢，需要不停迭代 latent space 去找合适的 latent code



### StyleGAN

> [blog](https://zhuanlan.zhihu.com/p/263554045)
> [Understanding Latent Space in Machine Learning](https://towardsdatascience.com/understanding-latent-space-in-machine-learning-de5a7c687d8d)

![](https://pic3.zhimg.com/80/v2-b54e4ac6af2ffb7e0b0b7697b64e937e_720w.webp)

- StyleGANv1, v2 差异

![](https://pic3.zhimg.com/80/v2-11a0432e1ebe5bc1c2de9b1e3575eaca_720w.webp)
![](https://pic2.zhimg.com/80/v2-acf5225a1534eadf19ead5c070014819_720w.webp)



### Referenced-based restoration

> This sensitivity may eventually lead to degraded results when the reference images/components are not well selected.

- "Blind face restoration via deep multi-scale component dictionaries" ECCV 2020
  [code](https://github.com/csxmli2016/DFDNet)

  > learnable dictionary
  >
  > computationally-intensive global matching [48] or component detection/selection

- "CrossNet: An end-to-end reference-based super resolution network using cross-scale warping" ECCV 2018

  >  single reference image

- "Image Super-Resolution by Neural Texture Transfer"
  [code](https://github.com/ZZUTK/SRNTT) [blog](https://zhuanlan.zhihu.com/p/464513517) [website](https://zzutk.github.io/SRNTT-Project-Page/)

  >  single reference image, **global matching**

- "Towards Content-Independent Multi-Reference Super-Resolution: Adaptive Pattern Matching and Feature Aggregation"
  [blog](https://zhuanlan.zhihu.com/p/262656707)

  > multiple reference images



## **methods**

![image-20230921164501263](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921164501263.png)

- Encoder

  提取各个尺度的特征 $f_i$ 。各层特征过一个 FC，得到 latent vector $C=\set{c_0, \cdots, c_k}$

  > convolutional features ${f_i}$ and the latent vectors $C$

  $c_i = E_{N+1}(f_N)$ 其中 $E_i , i \in  \{1, · · · , N\}$, denotes a stack of a stride-2 & stride-1 convolution

- Generative Latent Bank

  ![image-20230921173426982](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921173426982.png)

  each $c_i$ corresponds to one latent vector 对应 `StyleGAN` 每个 style block 的 A，Encoder 卷积层的特征 $f_i$ 代替 `StyleBlock` 的 Noise B

  输出 Style GAN 特征 $g_i$

  > 类似 Referenced-based restoration 中 learnable dictionary，区别在于 StyleGAN latent space is like a dictionary with potentially unlimited size and diversity
  >
  > **keep the weights of the latent bank fixed throughout training** 防止 dictionary 被 bias

  

- Decoder 融合 StyleGAN 特征 & Encoder 初始特征

  ![image-20230921173526218](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921173526218.png)

- Loss

  > adopt standard MSE loss, perceptual loss [23], and adversarial loss for training

  $$
  \mathcal{L}_{g}=\mathcal{L}_{mse}+\alpha_{percep}\cdot\mathcal{L}_{percep}+\alpha_{gen}\cdot\mathcal{L}_{gen}\\
  \text{where $\alpha_{percep}=\alpha_{gen}=10^{-2}$}
  $$

### Light-GLEAN

![image-20230921175254661](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921175254661.png)

> propose the following two strategies to simplify the structure of GLEAN. **LightGLEAN has only 21% of parameters when compared to GLEAN**
>
> 参考实现模型轻量化的方法

- remove the coarse feature connections between the encoder and the generator

  只保留 encoder 输出 $f_0$ 特征

- StyleGAN latent code 替换成可学习的 Parameters，而不是根据 Encoder 输出得到





## **Experiment**

> ablation study 看那个模块有效，总结一下

- Class-specific
  16x SR 人脸 和 其他数据，ESRGAN 生成细节不行

  ![image-20230921175931935](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921175931935.png)

  ![image-20230921182204167](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921182204167.png)

  - Cosine similarity

  ![image-20230921180039404](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921180039404.png)

  - multi-class SR

  先验替换 multi-class 先验 BigGAN，GLEAN outperforms existing works in terms of both fidelity and quality

  ![image-20230921180656729](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921180656729.png)

  ![image-20230921184527018](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921184527018.png)



- 其他 task

  -  Image colorization

    ![image-20230921180842097](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921180842097.png)

  - unknown degradations

    ![image-20230921181132282](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921181132282.png)

### Ablation Study

验证了模型主要的 3 个模块各自的效果

- Multi-resolution Encoder Features

  **做重建测试，去掉了 Decoder，用 generator（先验）直接输出结果**

  StyleGAN 加入 Encoder 多尺度的特征，输出的图像逐渐能够学到更为细节的信息。只用 Encoder 单个尺度特征，只能获取到全局信息，但细节差异很大

  ![image-20230921183119146](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921183119146.png)

- Latent Bank 有效性

  使用 prior 减轻 Encoder 负担，获取额外的细节信息，生成高质量 & 高保真度的结果

  ![image-20230921183341718](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921183341718.png)

- Decoder 的有效性

- 与 Reference-based 方法比较

  DFDNet 构造一个 eyes, lips 的 dictionary。对于 dictionary 里面有的特征修复的很好，但对于 dictionary  没有的特征，例如皮肤、头发，修复的就很差

  ![image-20230921184239242](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921184239242.png)



## **Limitations**

- 对于真实图像中的极端退化，修复效果仍然有限。

  ![image-20230921184721059](C:\Users\Loki\AppData\Roaming\Typora\typora-user-images\image-20230921184721059.png)



## **Summary :star2:**

> learn what & how to apply to our task

- details

  - Decoder 用 Encoder 浅层特征作为初始输入

  - Encoder 的特征过一下 Fully-Connected layer 更新以后得到 latent code. **类似 StyleGAN 方式**

    :question: 具体为啥要过 FC 看 StyleGAN ?

  - Perceptual loss, MSE, GAN loss 权重配置

    以 MSE loss 为主！

- idea

  GLEAN 用 StyleGAN 当作 **learnable dictionary** 可以与 reference based 联系起来。StyleGAN 当作一个无限大的 dictionary，但是不是很有针对性，与 reference images 结合一下？**用 reference images 过 Stable Diffusion 优化一下**

  - 换成 Stable Diffusion 