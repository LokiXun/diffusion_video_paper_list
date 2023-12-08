# DCNv2

> "Deformable ConvNets v2: More Deformable, Better Results" CVPR, 2018 Nov, **DCNv2** :statue_of_liberty:
> [paper](https://arxiv.org/abs/1811.11168)
> [paper local pdf](./2018_11_CVPR_Deformable-ConvNets-v2--More-Deformable--Better-Results.pdf)

## Keypoint

- Contribution

  1. more convolutional layers

     > deformable convolutions are applied in all the 3 × 3 conv layers in stages conv3, conv4, and conv5 in ResNet-50. Thus, there are 12 layers of deformable conv
     >
     > 对比 DCNv1，只有3层 Deformable Convolution

     Res101 Backbone, Param $ DCNv1~45.6 M \to +3.4M$，FLOPS+=0.7G

  2. modulation mechanism

     对输入特征图 x 中当前考虑的点 $p$，去采样附近 $K=k \times k$ 区域的点在 x 中特征，加权融合起来作为 $p$ 点的特征 y。在 paper 中设置 $k=3$
     $$
     y(p) = \sum_{k=1}^{K}{w_k}\cdot x(p+p_k+\triangle{p_k})\cdot \triangle{m_k} \\\\
     
     \text{ps: for each point $k$ in K points regions.} \\
     \text{1. $p_k$ is the point k's pre-specified location offset relative to p}\\
     \text{and $w_k$ is the weight of point k.}\\
     \text{2. $\triangle{p_k}$ is point k's \textbf{predicted offset} and $\triangle{m_k}$}\\
     \text{$\triangle{m_k}$ is the modulation sclar(also predicted), understand as another weight}
     $$

- 从可视化看

  1. 输入图像越小，感受也越大，能够捕捉到更完整的物体，**物体上的点更为密集** & 但会存在很多 offset 较大差异，到其他物体上的点

     随着输入图尺度增加，点会更分散 & 更为准确，偏移错的点比例减少

     从 ROI 区域来看，DCNv2 还是有偏差，**说明要按相似度再筛选一下，只加权还是有可能把错的信息拉进来**

     ![image-20230927120334217](C:\Users\Loki\workspace\Tongji_CV_group\docs\2018_11_CVPR_Deformable-ConvNets-v2--More-Deformable--Better-Results_Note\image-20230927120334217.png)

  2. 

