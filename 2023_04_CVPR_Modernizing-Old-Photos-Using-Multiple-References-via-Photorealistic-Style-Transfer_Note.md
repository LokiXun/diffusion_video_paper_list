# MROPM

> "Modernizing Old Photos Using Multiple References via Photorealistic Style Transfer" CVPR, 2023 Apr, **MROPM**
> [paper](https://arxiv.org/abs/2304.04461) [code](https://github.com/KAIST-VICLab/old-photo-modernization) [website](https://kaist-viclab.github.io/old-photo-modernization/?utm_source=catalyzex.com)
> [local paper pdf](./2023_04_CVPR_Modernizing-Old-Photos-Using-Multiple-References-via-Photorealistic-Style-Transfer.pdf)

## **Key-point**

> [website](https://kaist-viclab.github.io/old-photo-modernization/?utm_source=catalyzex.com) 里面的 key-idea 很清晰

之前 photo modernization 方法， "Bringing Old Photos Back to Life" 先 restore mixed degradation，再隐式地修复颜色 implicit enhance color。作者发现之增强颜色的结果，看起来的图像仍然很老式，因此作者想**将其它 reference 的风格迁移过去**使得图像看起来更 modern，**从风格角度进行修复**
老图像数据是作者从 3 个韩国博物馆获取到的一些文物图，噪声量很少，最多是有 blur 的情况，但看起来风格很老旧。



- **MROPM-Net** to modernize old color photos by **changing their styles** and enhancing them to look modern
  ![](https://kaist-viclab.github.io/old-photo-modernization/static/images/key_ideas/mropm_mropmnet.png)

- Stylization Subnet

  ![](https://kaist-viclab.github.io/old-photo-modernization/static/images/key_ideas/mropmnet_single.png)

  对 reference 分解为 local style code & global style code. 

  1. local style 与 old photo 用 no-local attention 对齐
  2. 对齐完的 local style 与 global 合并，用 AdaIN 去调整 old photo 的 mean, std 实现风格迁移

- merging-refinement subnet

  select the most appropriate styles from multiple stylized features for each semantic region in the old photo.

  融合多个迁移完风格的 old-photos. 对于每个区域，从多个迁移结果中选取最合适的。**从 style code 学习的 local style 的关联矩阵，做 attention 后作为权值与 style-code 相乘。**

  

  ![](https://kaist-viclab.github.io/old-photo-modernization/static/images/key_ideas/mropmnet_merging.png)



### Synthetic Data Generation Scheme

> 类似 Blind flickering 文章里面训 filter4

they propose a data generation scheme to **train the network in a self-supervised manner.** :star: The core idea is to use transformation with style-invariant (SIT) and variant (SVT) properties, determined by whether the transformation affects the mean and std of any semantic regions.

老照片没有 GT 数据，怎么训这个风格迁移网络呢？》》造老风格的数据，把一张图拆解为两个 reference，把主要内容颜色改掉作为 old photo

![](https://kaist-viclab.github.io/old-photo-modernization/static/images/key_ideas/mropm_synthetic_data.png)

- random color jittering and unstructured degradation, i.e., blur, noise, resizing, and compression artifacts are used for Style Variant Transformation (SVT)

  SVT 里面有 degradation



### Dataset

- 韩国 3 个博物馆收集到的文物照片，拍摄样式老旧，但没有明显的划痕

  从风格迁移的角度做

- License

  > [CC BY-NC-SA 4.0 details](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)

  ```
  All old photos in this dataset are licensed 
  under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International 
  (CC BY-NC-SA 4.0) https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
  ```

  



**Contributions**

## **Related Work**

- PST(photo realistic style transfer)

- AdaIN :question:

  AdaIN computes the style code (channel-wise mean and standard deviation) and uses it to align the mean and std of the target image. 

- SIT, SVT ??



## **methods**

**Limitations**





**Summary :star2:**

> learn what & how to apply to our task