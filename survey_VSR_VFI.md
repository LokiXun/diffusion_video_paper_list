# Survey on Video SR & VFI & deblur

> main note [README](./README.md)

- Q：diffusion Venhancer 生成的细节改变了，非 diffusion 方案生成人体形变比 diffusion 更严重

思考大致方向

- Reference based? 强调一致性
- 跨越 clip





## all-in-focus

- "Foreground-background separation and deblurring super-resolution method[☆](https://www.sciencedirect.com/science/article/pii/S0143816624006079#aep-article-footnote-id1)"

- "BokehMe: When Neural Rendering Meets Classical Rendering" CVPR-oral, 2022 Jun 25
  [paper](http://arxiv.org/abs/2206.12614v1) [code]() [pdf](./2022_06_CVPR-oral_BokehMe--When-Neural-Rendering-Meets-Classical-Rendering.pdf) [note](./2022_06_CVPR-oral_BokehMe--When-Neural-Rendering-Meets-Classical-Rendering_Note.md)
  Authors: Juewen Peng, Zhiguo Cao, Xianrui Luo, Hao Lu, Ke Xian, Jianming Zhang

![fig12](docs/2022_06_CVPR-oral_BokehMe--When-Neural-Rendering-Meets-Classical-Rendering_Note/fig12.png)

render bokeh effect 光圈虚化效果，需要给定 disparity 图（类似深度图）



- "BokehMe++: Harmonious Fusion of Classical and Neural Rendering for Versatile Bokeh Creation"





## Video SR

- "EDVR: Video Restoration with Enhanced Deformable Convolutional Networks" CVPR NTIRE 1st, 2019 May
  [paper](http://arxiv.org/abs/1905.02716v1) [code](https://github.com/xinntao/EDVR) [pdf](./2019_05_CVPR-NTIRE_EDVR--Video-Restoration-with-Enhanced-Deformable-Convolutional-Networks.pdf) [note](./2019_05_CVPR-NTIRE_EDVR--Video-Restoration-with-Enhanced-Deformable-Convolutional-Networks_Note.md)
  Authors: Xintao Wang, Kelvin C. K. Chan, Ke Yu, Chao Dong, Chen Change Loy

![image-20231115205018174](docs/2019_05_CVPR-NTIRE_EDVR--Video-Restoration-with-Enhanced-Deformable-Convolutional-Networks_Note/EDCR_framework.png)





- "BasicVSR++: Improving video super-resolution with enhanced propagation and alignment" CVPR, 2021 Apr :moyai:
  [paper](https://arxiv.org/abs/2104.13371) [code](https://github.com/open-mmlab/mmagic/blob/main/configs/basicvsr_pp/README.md) [note](./2021_04_CVPR_BasicVSR++--Improving-Video-Super-Resolution-with-Enhanced-Propagation-and-Alignment_Note.md)

1. Flow guided Deformable Transformer
2. 增加 second order residual 信息

![image-20231121170824361](docs/2021_04_CVPR_BasicVSR++--Improving-Video-Super-Resolution-with-Enhanced-Propagation-and-Alignment_Note/image-20231121170824361.png)





- "Investigating Tradeoffs in Real-World Video Super-Resolution" CVPR, 2021 Nov, **RealBasicVSR**
  [paper](https://arxiv.org/abs/2111.12704) [code](https://github.com/ckkelvinchan/RealBasicVSR) [note](./2021_11_CVPR_Investigating-Tradeoffs-in-Real-World-Video-Super-Resolution_Note.md)

![fig3](docs/2021_11_CVPR_Investigating-Tradeoffs-in-Real-World-Video-Super-Resolution_Note/fig3.png)

盲视频超分，**基于2个发现进行改进**：长时序反而会降低性能，有噪声没有特殊处理；iteration L=10 太少了会造成颜色伪影，20->30 会好一些；基于 BasicVSR 加入动态**预处理模块**，改进训练数据策略降低计算量





- "Recurrent Video Restoration Transformer with Guided Deformable Attention" NeurlPS, 2022 June, **RVRT** :statue_of_liberty:
  [paper](https://arxiv.org/abs/2206.02146) [code](https://github.com/JingyunLiang/RVRT) [note](./2022_06_NeurIPS_RVRT_Recurrent-Video-Restoration-Transformer-with-Guided-Deformable-Attention_Note.md)

![RVRT_Framework.png](docs/2022_06_NeurIPS_RVRT_Recurrent-Video-Restoration-Transformer-with-Guided-Deformable-Attention_Note/RVRT_Framework.png)





- "Rethinking Alignment in Video Super-Resolution Transformers" NIPS, 2022 Jul,`PSRT` :star:
  [paper](https://arxiv.org/abs/2207.08494) [code](https://github.com/XPixelGroup/RethinkVSRAlignment) [note](./2022_07_NIPS_Rethinking-Alignment-in-Video-Super-Resolution-Transformers_Note.md)

发现光流 warp 不适合 VSR 任务，光流存在很多噪声，**改成用 attention 去做对齐**





- "STDAN: Deformable Attention Network for Space-Time Video Super-Resolution" NNLS, 2023 Feb
  [paper](https://ieeexplore.ieee.org/document/10045744) [code](https://github.com/littlewhitesea/STDAN) [note](./2023_02_NNLS_STDAN--Deformable-Attention-Network-for-Space-Time-Video-Super-Resolution_Note.md)

![fig5](docs/2023_02_NNLS_STDAN--Deformable-Attention-Network-for-Space-Time-Video-Super-Resolution_Note/fig5.png)

Deformable Transformer 用到 video 上面，逐帧搞 deformable





- "Expanding Synthetic Real-World Degradations for Blind Video Super Resolution" CVPR, 2023 May
  [paper](https://arxiv.org/abs/2305.02660)





- "Enhancing Video Super-Resolution via Implicit Resampling-based Alignment" CVPR, 2023 Apr 29
  [paper](http://arxiv.org/abs/2305.00163v2) [code]() [pdf](./2023_04_CVPR_Enhancing-Video-Super-Resolution-via-Implicit-Resampling-based-Alignment.pdf) [note](./2023_04_CVPR_Enhancing-Video-Super-Resolution-via-Implicit-Resampling-based-Alignment_Note.md)
  Authors: Kai Xu, Ziwei Yu, Xin Wang, Michael Bi Mi, Angela Yao

![fig1](docs/2023_04_CVPR_Enhancing-Video-Super-Resolution-via-Implicit-Resampling-based-Alignment_Note/fig1.png)

发现 optical sample 中 bilinear 存在缺陷，**提出 implicit resample，改进 bilinear 采样方式**





- "Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution" ECCV, 2023 Dec, `MGLD-VSR`
  [paper](http://arxiv.org/abs/2312.00853v1) [code](https://github.com/IanYeung/MGLD-VSR) [note](2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution_Note.md) [pdf](./2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution.pdf)
  Authors: Xi Yang, Chenhang He, Jianqi Ma, Lei Zhang

![image-20240222173628376](docs/2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution_Note/image-20240222173628376.png)





- "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution" CVPR, 2023 Dec, `Upscale-A-Video`
  [paper](http://arxiv.org/abs/2312.06640v1) [code](https://github.com/sczhou/Upscale-A-Video) [website](https://shangchenzhou.com/projects/upscale-a-video/) [pdf](./2023_12_CVPR_Upscale-A-Video--Temporal-Consistent-Diffusion-Model-for-Real-World-Video-Super-Resolution.pdf)
  Authors: Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, Chen Change Loy

![image-20231220135955447](docs/2023_12_CVPR_Upscale-A-Video--Temporal-Consistent-Diffusion-Model-for-Real-World-Video-Super-Resolution_Note/Upscale-A-Video_framework.png)





- "TMP: Temporal Motion Propagation for Online Video Super-Resolution" TIP, 2023 Dec 15
  [paper](http://arxiv.org/abs/2312.09909v2) [code](https://github.com/xtudbxk/TMP.) [pdf](./2023_12_TIP_TMP--Temporal-Motion-Propagation-for-Online-Video-Super-Resolution.pdf) [note](./2023_12_TIP_TMP--Temporal-Motion-Propagation-for-Online-Video-Super-Resolution_Note.md)
  Authors: Zhengqiang Zhang, Ruihuang Li, Shi Guo, Yang Cao, Lei Zhang





- "FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring" CVPR-oral, 2024 Jan 8
  [paper](http://arxiv.org/abs/2401.03707v2) [code](https://kaist-viclab.github.io/fmanet-site) [pdf](./2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring.pdf) [note](./2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring_Note.md)
  Authors: Geunhyuk Youk, Jihyong Oh, Munchurl Kim

![fig3](docs/2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring_Note/fig3.png)

- 同时做 deblur + x4 SR；

- 实验发现 Dynamic Filter 传统 conv2d 版本不支持 large motion. 把输入特征加光流 warp 了一下，支持 large motion。。

  - 和  DCN 很类似，效果好一丢丢

- 训练数据使用 REDS 除去 REDS4 的 4 个视频。

- 对比 4 个方法 ok 了

  ![tb2](docs/2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring_Note/tb2.png)





- "Video Super-Resolution Transformer with Masked Inter&Intra-Frame Attention" CVPR, 2024 Jan 12
  [paper](http://arxiv.org/abs/2401.06312v4) [code](https://github.com/LabShuHangGU/MIA-VSR.) [pdf](./2024_01_CVPR_Video-Super-Resolution-Transformer-with-Masked-Inter&Intra-Frame-Attention.pdf) [note](./2024_01_CVPR_Video-Super-Resolution-Transformer-with-Masked-Inter&Intra-Frame-Attention_Note.md)
  Authors: Xingyu Zhou, Leheng Zhang, Xiaorui Zhao, Keze Wang, Leida Li, Shuhang Gu





- "CoCoCo: Improving Text-Guided Video Inpainting for Better Consistency, Controllability and Compatibility" Arxiv, 2024 Mar 18
  [paper](http://arxiv.org/abs/2403.12035v1) [code]() [pdf](./2024_03_Arxiv_CoCoCo--Improving-Text-Guided-Video-Inpainting-for-Better-Consistency--Controllability-and-Compatibility.pdf) [note](./2024_03_Arxiv_CoCoCo--Improving-Text-Guided-Video-Inpainting-for-Better-Consistency--Controllability-and-Compatibility_Note.md)
  Authors: Bojia Zi, Shihao Zhao, Xianbiao Qi, Jianan Wang, Yukai Shi, Qianyu Chen, Bin Liang, Kam-Fai Wong, Lei Zhang





- "Learning Spatial Adaptation and Temporal Coherence in Diffusion Models for Video Super-Resolution" CVPR, 2024 Mar 25
  [paper](http://arxiv.org/abs/2403.17000v1) [code]() [pdf](./2024_03_CVPR_Learning-Spatial-Adaptation-and-Temporal-Coherence-in-Diffusion-Models-for-Video-Super-Resolution.pdf) [note](./2024_03_CVPR_Learning-Spatial-Adaptation-and-Temporal-Coherence-in-Diffusion-Models-for-Video-Super-Resolution_Note.md)
  Authors: Zhikai Chen, Fuchen Long, Zhaofan Qiu, Ting Yao, Wengang Zhou, Jiebo Luo, Tao Mei





- "VideoGigaGAN: Towards Detail-rich Video Super-Resolution" ECCV, 2024 Apr 18 
  [paper](http://arxiv.org/abs/2404.12388v2) [code](https://github.com/danaigc/videoGigaGanHub) :warning: [web](https://videogigagan.github.io/) [pdf](./2024_04_ECCV_VideoGigaGAN--Towards-Detail-rich-Video-Super-Resolution.pdf) [note](./2024_04_ECCV_VideoGigaGAN--Towards-Detail-rich-Video-Super-Resolution_Note.md)
  Authors: Yiran Xu, Taesung Park, Richard Zhang, Yang Zhou, Eli Shechtman, Feng Liu, Jia-Bin Huang, Difan Liu(Adobe)

![fig3](docs/2024_04_ECCV_VideoGigaGAN--Towards-Detail-rich-Video-Super-Resolution_Note/fig3.png)

把 Image GigaGAN (未开源) 改到 Video 上面，加 temporal attention & 光流；把 downsample block 改为 Pool 降低伪影；只比较了 PSNR（没 BasicVSR++好）LPIPS(好了一些)，FVD





- "DaBiT: Depth and Blur informed Transformer for Joint Refocusing and Super-Resolution" Arxiv, 2024 Jul 1
  [paper](http://arxiv.org/abs/2407.01230v2) [code](https://github.com/crispianm/DaBiT) [pdf](./2024_07_Arxiv_DaBiT--Depth-and-Blur-informed-Transformer-for-Joint-Refocusing-and-Super-Resolution.pdf) [note](./2024_07_Arxiv_DaBiT--Depth-and-Blur-informed-Transformer-for-Joint-Refocusing-and-Super-Resolution_Note.md)
  Authors: Crispian Morris, Nantheera Anantrasirichai, Fan Zhang, David Bull





- "Arbitrary-Scale Video Super-Resolution with Structural and Textural Priors" ECCV, 2024 Jul 13
  [paper](http://arxiv.org/abs/2407.09919v1) [code](https://github.com/shangwei5/ST-AVSR.) [pdf](./2024_07_ECCV_Arbitrary-Scale-Video-Super-Resolution-with-Structural-and-Textural-Priors.pdf) [note](./2024_07_ECCV_Arbitrary-Scale-Video-Super-Resolution-with-Structural-and-Textural-Priors_Note.md)
  Authors: Wei Shang, Dongwei Ren, Wanying Zhang, Yuming Fang, Wangmeng Zuo, Kede Ma





- "RealViformer: Investigating Attention for Real-World Video Super-Resolution" ECCV, 2024 Jul 19
  [paper](http://arxiv.org/abs/2407.13987v1) [code](https://github.com/Yuehan717/RealViformer.) [pdf](./2024_07_ECCV_RealViformer--Investigating-Attention-for-Real-World-Video-Super-Resolution.pdf) [note](./2024_07_ECCV_RealViformer--Investigating-Attention-for-Real-World-Video-Super-Resolution_Note.md)
  Authors: Yuehan Zhang, Angela Yao





- "SeeClear: Semantic Distillation Enhances Pixel Condensation for Video Super-Resolution" NIPS, 2024 Oct 8
  [paper](http://arxiv.org/abs/2410.05799v4) [code](https://github.com/Tang1705/SeeClear-NeurIPS24) [pdf](./2024_10_NIPS_SeeClear--Semantic-Distillation-Enhances-Pixel-Condensation-for-Video-Super-Resolution.pdf) [note](./2024_10_NIPS_SeeClear--Semantic-Distillation-Enhances-Pixel-Condensation-for-Video-Super-Resolution_Note.md) :warning: 
  Authors: Qi Tang, Yao Zhao, Meiqin Liu, Chao Yao





- "Inflation with Diffusion: Efficient Temporal Adaptation for Text-to-Video Super-Resolution" WACV
  [paper](https://openaccess.thecvf.com/content/WACV2024W/VAQ/papers/Yuan_Inflation_With_Diffusion_Efficient_Temporal_Adaptation_for_Text-to-Video_Super-Resolution_WACVW_2024_paper.pdf)





### diffusion

- "Motion-Guided Latent Diffusion for Temporally Consistent Real-world Video Super-resolution" ECCV, 2023 Dec, `MGLD-VSR`
  [paper](http://arxiv.org/abs/2312.00853v1) [code](https://github.com/IanYeung/MGLD-VSR) [note](2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution_Note.md) [pdf](./2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution.pdf)
  Authors: Xi Yang, Chenhang He, Jianqi Ma, Lei Zhang

![image-20240222173628376](docs/2023_12_Arxiv_Motion-Guided-Latent-Diffusion-for-Temporally-Consistent-Real-world-Video-Super-resolution_Note/image-20240222173628376.png)





- "Upscale-A-Video: Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution" CVPR, 2023 Dec, `Upscale-A-Video`
  [paper](http://arxiv.org/abs/2312.06640v1) [code](https://github.com/sczhou/Upscale-A-Video) [website](https://shangchenzhou.com/projects/upscale-a-video/) [pdf](./2023_12_CVPR_Upscale-A-Video--Temporal-Consistent-Diffusion-Model-for-Real-World-Video-Super-Resolution.pdf)
  Authors: Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, Chen Change Loy

![image-20231220135955447](docs/2023_12_CVPR_Upscale-A-Video--Temporal-Consistent-Diffusion-Model-for-Real-World-Video-Super-Resolution_Note/Upscale-A-Video_framework.png)





### cartoon

- "AnimeSR: Learning Real-World Super-Resolution Models for Animation Videos" NIPS, 2022 Jul :star:
  [paper](https://arxiv.org/abs/2206.07038) [code](https://github.com/TencentARC/AnimeSR#open_book-animesr-learning-real-world-super-resolution-models-for-animation-videos)





### Device Efficiency

- "Towards High-Quality and Efficient Video Super-Resolution via Spatial-Temporal Data Overfitting" CVPR_highlight, 2023 Mar
  [paper](http://arxiv.org/abs/2303.08331v2) [code](https://github.com/coulsonlee/STDO-CVPR2023.git) [note](./2023_03_CVPR_highlight_Towards-High-Quality-and-Efficient-Video-Super-Resolution-via-Spatial-Temporal-Data-Overfitting_Note.md)



### Event Camera

- "EvTexture: Event-driven Texture Enhancement for Video Super-Resolution" Arxiv, 2024 Jun 19, `EvTexture`
  [paper](http://arxiv.org/abs/2406.13457v1) [code](https://github.com/DachunKai/EvTexture) [pdf](./2024_06_Arxiv_EvTexture--Event-driven-Texture-Enhancement-for-Video-Super-Resolution.pdf) [note](./2024_06_Arxiv_EvTexture--Event-driven-Texture-Enhancement-for-Video-Super-Resolution_Note.md)
  Authors: Dachun Kai, Jiayao Lu, Yueyi Zhang, Xiaoyan Sun





### 3D SR :bear:

- "SuperGaussian: Repurposing Video Models for 3D Super Resolution" ECCV, 2024 Jun 2
  [paper](http://arxiv.org/abs/2406.00609v4) [code](https://github.com/adobe-research/SuperGaussian) [pdf](./2024_06_ECCV_SuperGaussian--Repurposing-Video-Models-for-3D-Super-Resolution.pdf) [note](./2024_06_ECCV_SuperGaussian--Repurposing-Video-Models-for-3D-Super-Resolution_Note.md)
  Authors: Yuan Shen, Duygu Ceylan, Paul Guerrero, Zexiang Xu, Niloy J. Mitra, Shenlong Wang, Anna Frühstück



- "Sequence Matters: Harnessing Video Models in 3D Super-Resolution" AAAI, 2024 Dec 16
  [paper](http://arxiv.org/abs/2412.11525v3) [code](https://ko-lani.github.io/Sequence-Matters) [pdf](./2024_12_AAAI_Sequence-Matters--Harnessing-Video-Models-in-3D-Super-Resolution.pdf) [note](./2024_12_AAAI_Sequence-Matters--Harnessing-Video-Models-in-3D-Super-Resolution_Note.md)
  Authors: Hyun-kyu Ko, Dongheok Park, Youngin Park, Byeonghyeon Lee, Juhee Han, Eunbyung Park







## Video Deblur

- "Spatio-Temporal Filter Adaptive Network for Video Deblurring" ICCV, 2019 
  [code](https://github.com/sczhou/STFAN)

Dynamic Filter Network 预测退化特征，融入 deblur/SR





- "High-resolution optical flow and frame-recurrent network for video super-resolution and deblurring" NeuralComputing, 2022 Jun 7, `HOFFR`
  [paper](https://www.sciencedirect.com/science/article/pii/S0925231222002363) [code]() [pdf](./2022_06_NeuralComputing_High-resolution-optical-flow-and-frame-recurrent-network-for-video-super-resolution-and-deblurring.pdf) [note](./2022_06_NeuralComputing_High-resolution-optical-flow-and-frame-recurrent-network-for-video-super-resolution-and-deblurring_Note.md)
  Authors: Ning Fang, Zongqian Zhan

双分支，一个搞 SR，一个 Deblur 最后 channel attention 合起来；一开始预测一个光流

![fig1](docs/2022_06_NeuralComputing_High-resolution-optical-flow-and-frame-recurrent-network-for-video-super-resolution-and-deblurring_Note/fig1.png)





- "Rethinking Video Deblurring with Wavelet-Aware Dynamic Transformer and Diffusion Model" ECCV, 2024 Aug 24
  [paper](http://arxiv.org/abs/2408.13459v1) [code](https://github.com/Chen-Rao/VD-Diff) [pdf](./2024_08_ECCV_Rethinking-Video-Deblurring-with-Wavelet-Aware-Dynamic-Transformer-and-Diffusion-Model.pdf) [note](./2024_08_ECCV_Rethinking-Video-Deblurring-with-Wavelet-Aware-Dynamic-Transformer-and-Diffusion-Model_Note.md)
  Authors: Chen Rao, Guangyuan Li, Zehua Lan, Jiakai Sun, Junsheng Luan, Wei Xing, Lei Zhao, Huaizhong Lin, Jianfeng Dong, Dalong Zhang

![fig1](docs/2024_08_ECCV_Rethinking-Video-Deblurring-with-Wavelet-Aware-Dynamic-Transformer-and-Diffusion-Model_Note/fig1.png)





### Dynamic Filter Network

理解为根据输入退化，动态地预测类似卷积核，效果比 DCN 好一些



- "Spatio-Temporal Filter Adaptive Network for Video Deblurring" ICCV, 2019 
  [code](https://github.com/sczhou/STFAN)



- "FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring" CVPR-oral, 2024 Jan 8
  [paper](http://arxiv.org/abs/2401.03707v2) [code](https://kaist-viclab.github.io/fmanet-site) [pdf](./2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring.pdf) [note](./2024_01_CVPR-oral_FMA-Net--Flow-Guided-Dynamic-Filtering-and-Iterative-Feature-Refinement-with-Multi-Attention-for-Joint-Video-Super-Resolution-and-Deblurring_Note.md)
  Authors: Geunhyuk Youk, Jihyong Oh, Munchurl Kim





## Space-Time VSR

VSR+VFI

- "How Video Super-Resolution and Frame Interpolation Mutually Benefit" ACMM 
  [paper](https://discovery.ucl.ac.uk/id/eprint/10136963/1/ChengchengZhou-ACMMM2021-final.pdf)



- "Enhancing Space-time Video Super-resolution via Spatial-temporal Feature Interaction" Arxiv, 2022 Jul

https://github.com/yuezijie/STINet-Space-time-Video-Super-resolution?tab=readme-ov-file



- "RSTT: Real-time Spatial Temporal Transformer for Space-Time Video Super-Resolution" CVPR 2022
  [paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Geng_RSTT_Real-Time_Spatial_Temporal_Transformer_for_Space-Time_Video_Super-Resolution_CVPR_2022_paper.pdf)



- "[Scale-adaptive feature aggregation for efficient space-time video super-resolution](https://openaccess.thecvf.com/content/WACV2024/html/Huang_Scale-Adaptive_Feature_Aggregation_for_Efficient_Space-Time_Video_Super-Resolution_WACV_2024_paper.html)" WACV 2024



- "Complementary Dual-Branch Network for Space-Time Video Super-Resolution" ICPR, 2024 Dec 05 

  https://link.springer.com/chapter/10.1007/978-3-031-78125-4_13



- "A Resource-Constrained Spatio-Temporal Super Resolution Model" 2024



- "Global Spatial-Temporal Information-based Residual ConvLSTM for Video Space-Time Super-Resolution" 2024 Jul 
  [paper](https://arxiv.org/pdf/2407.08466)



- "VEnhancer: Generative Space-Time Enhancement for Video Generation" Arxiv, 2024 Jul 10
  [paper](http://arxiv.org/abs/2407.07667v1) [code]() [pdf](../2024_07_Arxiv_VEnhancer--Generative-Space-Time-Enhancement-for-Video-Generation.pdf) [note](../2024_07_Arxiv_VEnhancer--Generative-Space-Time-Enhancement-for-Video-Generation_Note.md)
  Authors: Jingwen He, Tianfan Xue, Dongyang Liu, Xinqi Lin, Peng Gao, Dahua Lin, Yu Qiao, Wanli Ouyang, Ziwei Liu



- "3DAttGAN: A 3D Attention-based Generative Adversarial Network for Joint Space-Time Video Super-Resolution" 
  [paper](https://github.com/FCongRui/3DAttGan/tree/main/Code)





## VFI

- "LDMVFI: Video Frame Interpolation with Latent Diffusion Models" Arxiv, 2023 Mar, `LDMVFI`
  [paper](https://arxiv.org/abs/2303.09508) [code](https://github.com/danier97/LDMVFI) [note](./2023_03_Arxiv_LDMVFI--Video-Frame-Interpolation-with-Latent-Diffusion-Models_Note.md)





- "Disentangled Motion Modeling for Video Frame Interpolation" Arxiv, 2024 Jun 25
  [paper](http://arxiv.org/abs/2406.17256v1) [code](https://github.com/JHLew/MoMo) [pdf](./2024_06_Arxiv_Disentangled-Motion-Modeling-for-Video-Frame-Interpolation.pdf) [note](./2024_06_Arxiv_Disentangled-Motion-Modeling-for-Video-Frame-Interpolation_Note.md)
  Authors: Jaihyun Lew, Jooyoung Choi, Chaehun Shin, Dahuin Jung, Sungroh Yoon

We use [Vimeo90k](http://toflow.csail.mit.edu/) for training, and use [SNU-FILM](https://myungsub.github.io/CAIN/), [Xiph](https://github.com/JHLew/MoMo/blob/main/dataset.py#L168), [Middlebury-others](https://vision.middlebury.edu/flow/data/) for validation.



- "VFIMamba: Video Frame Interpolation with State Space Models" NIPS, 2024 Oct

  https://github.com/MCG-NJU/VFIMamba



- "Perception-Oriented Video Frame Interpolation via Asymmetric Blending" CVPR, 2024 Apr 10, `PerVFI`
  [paper](http://arxiv.org/abs/2404.06692v1) [code](https://github.com/mulns/PerVFI) [pdf](./2024_04_CVPR_Perception-Oriented-Video-Frame-Interpolation-via-Asymmetric-Blending.pdf) [note](../2024_04_CVPR_Perception-Oriented-Video-Frame-Interpolation-via-Asymmetric-Blending_Note.md)
  Authors: Guangyang Wu, Xin Tao, Changlin Li, Wenyi Wang, Xiaohong Liu, Qingqing Zheng





### Image

- "DynamiCrafter: Animating Open-domain Images with Video Diffusion Priors" ECCV, 2023 Oct 18
  [paper](http://arxiv.org/abs/2310.12190v2) [code]() [pdf](./2023_10_ECCV_DynamiCrafter--Animating-Open-domain-Images-with-Video-Diffusion-Priors.pdf) [note](./2023_10_ECCV_DynamiCrafter--Animating-Open-domain-Images-with-Video-Diffusion-Priors_Note.md)
  Authors: Jinbo Xing, Menghan Xia, Yong Zhang, Haoxin Chen, Wangbo Yu, Hanyuan Liu, Xintao Wang, Tien-Tsin Wong, Ying Shan



- "ToonCrafter: Generative Cartoon Interpolation" Arxiv, 2024 May 28
  [paper](http://arxiv.org/abs/2405.17933v1) [code]() [pdf](./2024_05_Arxiv_ToonCrafter--Generative-Cartoon-Interpolation.pdf) [note](./2024_05_Arxiv_ToonCrafter--Generative-Cartoon-Interpolation_Note.md)
  Authors: Jinbo Xing, Hanyuan Liu, Menghan Xia, Yong Zhang, Xintao Wang, Ying Shan, Tien-Tsin Wong



- Film: Frame interpolation for large motion.



### cartoon

- "Thin-Plate Spline-based Interpolation for Animation Line Inbetweening" 
  [paper](https://arxiv.org/pdf/2408.09131)





## AnyResolution(THW)

- "Lumina-T2X: Transforming Text into Any Modality, Resolution, and Duration via Flow-based Large Diffusion Transformers" Arxiv, 2024 May 9
  [paper](http://arxiv.org/abs/2405.05945v3) [code]() [pdf](./2024_05_Arxiv_Lumina-T2X--Transforming-Text-into-Any-Modality--Resolution--and-Duration-via-Flow-based-Large-Diffusion-Transformers.pdf) [note](./2024_05_Arxiv_Lumina-T2X--Transforming-Text-into-Any-Modality--Resolution--and-Duration-via-Flow-based-Large-Diffusion-Transformers_Note.md)
  Authors: Peng Gao, Le Zhuo, Dongyang Liu, Ruoyi Du, Xu Luo, Longtian Qiu, Yuhang Zhang, Chen Lin, Rongjie Huang, Shijie Geng, Renrui Zhang, Junlin Xi, Wenqi Shao, Zhengkai Jiang, Tianshuo Yang, Weicai Ye, He Tong, Jingwen He, Yu Qiao, Hongsheng Li





## VAE 优化

- "CV-VAE: A Compatible Video VAE for Latent Generative Video Models" NIPS, 2024 May 30
  [paper](http://arxiv.org/abs/2405.20279v2) [code](https://github.com/AILab-CVC/CV-VAE) [pdf](./2024_05_NIPS_CV-VAE--A-Compatible-Video-VAE-for-Latent-Generative-Video-Models.pdf) [note](./2024_05_NIPS_CV-VAE--A-Compatible-Video-VAE-for-Latent-Generative-Video-Models_Note.md)
  Authors: Sijie Zhao, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Muyao Niu, Xiaoyu Li, Wenbo Hu, Ying Shan

SVD 优化 3D VAE 25frames-> 96frames





## Text/FrameConsistency

- "VideoRepair: Improving Text-to-Video Generation via Misalignment Evaluation and Localized Refinement" Arxiv, 2024 Nov 22
  [paper](http://arxiv.org/abs/2411.15115v1) [code]() [pdf](./2024_11_Arxiv_VideoRepair--Improving-Text-to-Video-Generation-via-Misalignment-Evaluation-and-Localized-Refinement.pdf) [note](./2024_11_Arxiv_VideoRepair--Improving-Text-to-Video-Generation-via-Misalignment-Evaluation-and-Localized-Refinement_Note.md)
  Authors: Daeun Lee, Jaehong Yoon, Jaemin Cho, Mohit Bansal



- "Video-Infinity: Distributed Long Video Generation"
