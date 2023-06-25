# Diffusion Video Autoencoders

> [2023_CVPR_Diffusion-Video-Autoencoders--Toward-Temporally-Consistent-Face-Video-Editing-via-Disentangled-Video-Encoding.pdf](./2023_CVPR_Diffusion-Video-Autoencoders--Toward-Temporally-Consistent-Face-Video-Editing-via-Disentangled-Video-Encoding.pdf)
> [code](https://github.com/man805/Diffusion-Video-Autoencoders)
>
> - 算力: 4*V100
> - [baseline_comparision 2021_ACM_Designing-an-Encoder-for-StyleGAN-Image-Manipulation.pdf](2021_ACM_Designing-an-Encoder-for-StyleGAN-Image-Manipulation.pdf)

**Background**

face video editing (edit hair color, gender, wearing glasses, etc.) bearing the challenge of *temporal consistency* among edited frames.

**Contributions**

1. **devise diffusion video autoencoders based on diffusion autoencoders that decompose the video into a single time-invariant and per-frame time-variant features** for temporally consistent editing.
1. Based on the decomposed representation of diffusion video autoencoder, face video editing can be conducted by **editing only the single time-invariant identity feature** and decoding it together with the remaining original features
1. our framework can be utilized to **edit exceptional cases**(partially occluded)
1. **text-based editing methods（CLIP-loss）**



**Related Work**

> - Diffusion Autoencoder

- Style-**GAN based methods** edit each frames, ***unable to perfectly reconstruct original image*** , especially when occlusion happens:question:

  1. 一些方法针对无法复原原图的问题，继续 finetune GAN-inversion。但会有失去原来可编辑能力的风险，针对视频多帧更为严重。

- video temporal consistency :question:

  对 latent trajectory or features 进性平滑，但无法保证一致性。是因为其隐式地改变了 motion feature



## methods

![DVA_overview.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\DVA_overview.jpg)

1. 模仿 [STIT](#STIT :star:) 对所有帧先 crop & align 

2. 对每帧提取 $Z_{face}^{(i)}$
   $$
   Z_{id\_rep} = 1/N * \sum_{n=1}^{N}{(Z_{id}^{(n)})}
   \\
   Z_{face}^{(n)} = MLP(Concatenate(Z_{id\_rep}, Z_{landscape}^{(n)}))
   $$

3. DDIM forward process conditioned on $Z_{face}$

4. 对 ID feature 进性编辑$Z_{id,rep}^{edit}$ ，之后跟原来一样得到 $Z_{face}^{(n),edit} = MLP(Concatenate(Z_{id,rep}^{edit}, Z_{landscape}^{(n)}))$

   - pretrained linear attribute-classifier
   - CLIP

5. DDIM reverse process 获取修改完对应画面

6. paste back: 用 BiSeNetV2 分割人脸



### Distangle Video Encoding

Distangle the video into $Z_{face} = MLP(Concatenate(Z_{id}, Z_{landscape}^{(n)}))$， $Z_{T}$ noise map for background information.

**Identity Feature**

1. $Z_{id}$ >> ArcFace get identity feature

   $Z_{id}$ regards as time-variant feature. So denote without frame index in the right-up corner.

   Id 特征在所有帧共享，但 ArcFace 提取出的任务类别特征每帧有差异，文中直接 average :rotating_light:

   > :question: average 如果对于人戴面具的场景？？

   

2. $Z_{landscape}^{(n)}$ >> [pytorch Facial Landmarks code](https://github.com/cunjian/pytorch_face_landmark) 提取人脸关键点，提取运动特征



为了提纯 $Z_{face}$ 使用两个 Loss objective $L_{DVA} = L_{simple} + L_{reg}$

1. DDPM loss
2. L_reg loss 采样两个 nosise 和 mask 相乘，提取面部区域计算 L1 loss

![DVA_overview.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\DVA_overview.jpg)

- ablation study for  Loss_reg

![DVA_ablation_Loss_reg.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\DVA_ablation_Loss_reg.jpg)



#### ArcFace

> [2022_ArcFace-Additive-Angular-Margin-Loss-for-DeepFace-Recognition.pdf](./FaceFeature/2022_ArcFace-Additive-Angular-Margin-Loss-for-DeepFace-Recognition.pdf)
> [2021_CVPR_One-Shot-Free-View-Neural-Talking-Head-Synthesis-for-Video-Conferencing.pdf](./FaceFeature/2021_CVPR_One-Shot-Free-View-Neural-Talking-Head-Synthesis-for-Video-Conferencing.pdf)

ArcFace比Softmax的特征分布更紧凑，决策边界更明显，一个弧长代表一个类。



#### Facial Landmarks

> [pytorch Facial Landmarks code](https://github.com/cunjian/pytorch_face_landmark) >> 面部关键点特征



### Editing :key:

**Classifier-based editing**

#### DiffAE

> [博客参考](https://zhuanlan.zhihu.com/p/625386246)
> [code](https://github.com/phizaz/diffae)



#### ProgressiveGrowingGAN

> - [Progressive Growing of GANs for Improved Quality, Stability, and Variation](https://arxiv.org/abs/1710.10196)
>   [code](https://github.com/tkarras/progressive_growing_of_gans.git)



**CLIP-based editing**

#### StyleGAN-NADA

> [StyleGAN-NADA: **CLIP-Guided** Domain Adaptation of Image Generators](https://arxiv.org/abs/2108.00946)
> [StyleGAN-NADA blog](https://zhuanlan.zhihu.com/p/422788325)



