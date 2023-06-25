# DISTS

> [2020_TPAMI_DISTS_Image-Quality-Assessment-Unifying-Structure-and-Texture-Similarity.pdf](./2020_TPAMI_DISTS_Image-Quality-Assessment-Unifying-Structure-and-Texture-Similarity.pdf)
>
> - 算法改进小结
>   1. 单独对于一个实验（quality prediction）效果不好别放弃，比较其他实验（texture similarity，retrieval && **可视化看效果更加直观**）
>   2. 找数据集上存在的问题，去想办法改进算法
> - 1*GTX2080

**Existing Problem**

- Full-reference IQA methods(SSIM) highly sensitive to same-textured images. 
  针对有明显纹理的原图，让模型对 JPEG 压缩后、resample 的图像打分（实际上肉眼看上去 JPEG 更加模糊），之前方法对于 JPEG 图像质量评分错误地高于 resample 图。

  - develop objective IQA metrics consistent with perceptual similarity

    图像压缩能优化成存储 texture 而不是还原全部原始 pixel

- DISTS 优化点

  1. 对 resampling visual texture 不敏感
  2. 更加符合 human quality judgement
  3. 在 texture classification/retrieval，insensitive to distortions



**Background**

- 3 种传统 IQA methods

  1. SSIM

     无法处理 visual texture

  2. knowledge-based IQA

     measure image quality based on pixel-by-pixel comparison，无法处理 visual texture

  3. data-driven IQA

     数据集少，容易过拟合

- computational texture features

  无法有效的和 human rating of texture similarity 关联



## **Methods**

- 设计了一个函数将 pixel representation 映射到更均匀分布地空间，因为要用 MSE（假定所有分布权重相同，pixel representation 各个像素权重明显不同）

- injective relationship

  将 representation 映射到唯一的 output，实现如果 representation of image 不同，对应地输出图一定不同（针对纹理图像改进）



