# Neural Matching Fields

> "Neural Matching Fields: Implicit Representation of Matching Fields for Visual Correspondence" NeurIPS, 2022 Oct
> [paper](https://arxiv.org/abs/2210.02689) [code](https://github.com/KU-CVLAB/NeMF/) [website](https://ku-cvlab.github.io/NeMF/)
> [paper local pdf](./2022_10_NeurIPS_Neural-Matching-Fields--Implicit-Representation-of-Matching-Fields-for-Visual-Correspondence.pdf)

## Key-point

- Task:  特征点匹配 semantic correspondence

- Background: See Section 4.1

  给定 source, target 图像对 $I_s, I_t \in \mathcal{R}^{H_t \times H_w}$，目标找一个 correspondence field $F(x)$ 表示 source 每个像素的偏移, $I_{t}(\mathbf{x})\approx I_{s}(\mathbf{x}+F(\mathbf{x}))$ 。参考先前方法，对图像提取特征 $D(X)$，计算成对的相似度 $C(x,y)$
  $$
  C(\mathbf{x},\mathbf{y})=\frac{D_s(\mathbf{x})\cdot D_t(\mathbf{y})}{\|D_s(\mathbf{x})\|\|D_t(\mathbf{y})\|}
  $$
  先前方法受到输入图尺寸的限制，提取特征对计算or存储的要求很高，因此用 CNN 深层的特征（ $x\in [0, h_s)\times [0, h_s), ~ h < H$）再去提取特征 $D(x)$ 的尺寸弄得很小，需要后续进行 up-sample 后处理，对应到高分辨率的 correspondence map。**这种方式提取到的 C(x,y) 很粗糙 & up-sample 存在损失**

首个将隐式网络用于特征对齐的工作，用 Cost Embedding Network 改进 INR 里面简单的 FC layer。

![](https://ku-cvlab.github.io/NeMF/resources/visualization3d.png)



## Contributions

- 首个将隐式网络用于特征对齐的工作，用 Cost Embedding Network 改进 INR 里面简单的 FC layer

  > INR has never been properly studied or explored in visual correspondence tasks

- 提出 PatchMatch + Coordinate Optimization 策略加速推理



## Related Work

- hand-crafted feature descriptor

  提取的特征 high-level 信息不足

- CNN 方法

  "Cats++: Boosting cost aggregation with convolutions and transformers"

  CNN 得到的特征，相比 SIFT 等人工设计的算子有更好的语义不变性，但得到的 correspondence map 分辨率低 （输入图像分辨率限制）& 还得用 hand-crafted 的 bilinear 插值方式 up-sample，存在损失

- coarse-to-fine approach

  "Deep matching prior: Test-time optimization for dense correspondence"

  多尺度融合，容易受初始阶段误差累计的影响；参考用于加速推理

- Implicit Neural Representation (INR)

  直接用 FC 太简单没法表示高维

  因此改进用 cost embedding network with convolutions 来表示 local 内容，self-attention layers 获取全局感受野



## methods

### Training

![](https://ku-cvlab.github.io/NeMF/resources/figure_3.png)

1. 图像中随机采样 x,y 得到 4D 坐标  $p=(x,y)$ ，x,y 分别为从 source & reference 图像提取 2D 坐标

   $\gamma(p) \text{ encoded point of p}$

2. 用预训练的 Res101 提取特征 $D(x)$，计算 Cost Volume $C(\mathbf{x},\mathbf{y})=\frac{D_s(\mathbf{x})\cdot D_t(\mathbf{y})}{\|D_s(\mathbf{x})\|\|D_t(\mathbf{y})\|}$

3. `Cost Embedding Network` 计算 cost vector $\phi{(C,p)} \text{ cost feature vector at p}$

4. 全连接组成的 $f_\theta(\gamma(p), ~\phi(C,p))$ 计算匹配图 M

   直接 concat 两个输入，输入到 MLP，里面 BatchNorm 很占显存！

   参考这个 "Differentiable Volumetric Rendering: Learning Implicit 3D Representations without 3D Supervision" 

- GT 关键点对 $\{\mathbf{x},\mathbf{x}^{\prime}\}, y=\mathbf{x}^{\prime}, ~p=[x,y]$,  希望网络输出 1，对于非 GT 点对输出 0

  非 GT 点对直接随机采样（均匀分布）

- Loss

  1. classification: cross-entropy loss
     $$
     \mathcal{L}_f=-\sum_{k=1}^KM_k^*\mathrm{log}(M(\mathbf{p}_k)),\\
     M_k^* =1 ~if~gt~else~0
     $$

  2. Flow loss 

     > :warning: 个人理解可能不对

     $\phi(C,\mathbf{p})\in\mathbb{R}^{H_{s}\times W_{s}\times H_{t}\times W_{t}\times K} $代表每个像素点之间的相似度。将模型预测的结果和 GT相似度矩阵，计算距离。

     
     $$
     V=\mathrm{avgpool}(\phi(C,\mathbf{p}))\in\mathbb{R}^{H_{s}\times W_{s}\times H_{t}\times W_{t}}\\
     F_{pred} = soft\_Argmax(V)\\
     $$
     Loss
     $$
     \mathcal{L}_c=\|F_{\mathrm{gt}}-F_{\mathrm{pred}}\|_2.
     $$

- 



#### Cost Embedding Network

- NeRF background
  $$
  \gamma(t)=[\sin(2^0t\pi),\cos(2^0t\pi),...,\sin(2^Lt\pi),\cos(2^Lt\pi)], ~\text{encoding function}\\
  \{\sigma,\mathbf{c}\}=f_\omega(\gamma(\mathbf{o}),\gamma(\mathbf{d}))
  $$

- Motivation

  假设直接用 $f_\theta(\gamma(p))$ 去训练 $f_\theta$ ，对于高维 or 复杂的 field 有点困难，训练当前这个 Cost Embedding Network 去提取特征，作为 guidance

将 Cost Volume $C(\mathbf{x},\mathbf{y})=\frac{D_s(\mathbf{x})\cdot D_t(\mathbf{y})}{\|D_s(\mathbf{x})\|\|D_t(\mathbf{y})\|}$ 提取特征转换为 $\phi{(C,p)}$ >> 用 Self-attn layer 提取特征，得到 5D 特征
$$
C^{\prime}\in\mathbb{R}^{H_s\times W_s\times H_t\times W_t\times K}
$$
对于 $p$ 点，用 quadlinear 插值获取  query point $\phi(C, \bold{p}) \in \mathbb{R}^{K\times 1}$

- **但缺少类似 relative positioning bias 的操作**

  > combine Transformer architecture [79] with convolution operator to compensate the lack of inductive bias,



### Inference

- 直接每个点输入去 query，遍历两张图每个点消耗太大 >> PatchMatch-based Sampling

  缺点：降低搜索范围

![](https://ku-cvlab.github.io/NeMF/resources/figure_4.png)



#### PatchMatch-based Sampling

对于查询点 x, 迭代 N 次去**找相似度最大的一些候选点集**。实际实现时候，对多个 query point 并行去迭代

- 每次迭代

  对于查询点 x，去到 $V=\mathrm{avgpool}(\phi(C,\mathbf{p}))\in\mathbb{R}^{H_{s}\times W_{s}\times H_{t}\times W_{t}}\\$ 里面采样最可能的点（论文说是 V，下面公式更能理解一些）
  $$
  F^l(\mathbf{x})=\operatorname{argmax}_{\mathbf{y}\in\mathcal{Y}^{l-1}}(M([\mathbf{x},\mathbf{y}]))\\
  \mathcal{Z}^{l-1}=\{F^{l-1}(\mathbf{z})\}_{\mathbf{z}}
  $$
  

  再加上一些随机采样点，组成当前迭代的候选点集 $\mathcal{Y}^{l-1}=\bigcup\left(\mathcal{Z}^{l-1},\{\mathbf{y}\}\right)$
  $$
  F^l(\mathbf{x})=\operatorname{argmax}_{\mathbf{y}\in\mathcal{Y}^{l-1}}(M([\mathbf{x},\mathbf{y}]))\\
  $$

- 缺点：候选点集降低了搜索范围，引入了误差，提出下面 Coordinate Optimization 去进一步修正



#### Coordinate Optimization

上面得到的候选点集虽然不是很准确，但也很接近。**用 Match score 的 log 值，相对于预测坐标 y 的梯度反传，去更新候选点集中每个点 y 的坐标**
$$
\begin{aligned}&
\mathcal{L}=-\log(M([\mathbf{x},\mathbf{y}])),\\&\mathbf{y}:=\mathbf{y}-\alpha\nabla_\mathbf{y}{\mathcal{L}},
\end{aligned}
$$


## Experiment

> ablation study 看那个模块有效，总结一下

- Seetting

  1. Feature-extractor
     pre-trained on ImageNet ResNet-101 去提取特征
  2. Cost Embdding Network 直接参考下面这个
     "CATs: Cost Aggregation Transformers for Visual Correspondence"
  3. 推理时的迭代次数，N = 10 for both PatchMatch and coordinate optimizations,
  4. feature maps resized to 16×16 for constructing a coarse cost volume

- Dataset: 特征点匹配数据

- Metrics
  $$
  percentage~of~correct~keypoints (PCK) = \\
  d(k_{\mathrm{pred}},k_{\mathrm{GT}})\leq\alpha\cdot\max(H,W)
  $$
  $d$ 为 L2 距离，$\alpha$ 为阈值（设置 0.01, 0.03, ...）



### Ablation Study

- Different Cost Feature Representation

  发现对计算得到的 Cost Volume，用 Conv + Self-attn 处理一下会好很多

- 推理策略

  在一对 source ref pair 上比较速度。

  每个点都搜一遍，> 300k s （83h）也就没测 PCK；用 corrdinate opt 进一步优化 PCK 接近 2 个点！

- 计算复杂度

  1. 推理一对图，8-9s

  2. 显存占用

     feature maps resized to 16×16 for constructing a coarse cost volume





## **Limitations**

- 推理一张图要 8-9s，如果对于 7 x 5 张（local+ref）要 35 * 10 要 6 min才能有匹配点的结果

  > N = 10, the time taken at the inference phase for a single sample is approximately 8-9 seconds on a single GPU Geforce RTX 3090





## **Summary :star2:**

> learn what & how to apply to our task
>
> video local 7 帧，去到 reference 5 帧里面提取想要的特征。先前实验发现随机取 reference 如果差异太大，训练会崩掉

- 每个 local 图像和 reference 图像获取 matching field 每个元素在 （0，1）范围，看作是相似度，去给 reference 加权，剔除掉差异太大的 reference & 提取细节？