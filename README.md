# **Tongji_CV_Journey** :gun:

> This document mainly works as an **paper List in categories** :cat:
> Also, our notes for read papers are linked beside, which could help us recall the main idea in paper more quickly.
>
> :ticket: Note that
>
> 1. The paper Information is listed at such format
>
>    ```
>    "Paper Name" Conference/Journal/Arxiv, year month, MethodsAbbreviation
>    Authors(optional)
>    [paper link]() [code link]() [paper website link]()
>    [the Note link, which we makde summary based on our understanding]()
>    short discription(optional)
>    ```
>
> 2. If only the paper website is listed, it denotes the paper link and code link could be found in the website page.
>
> 3. The priority order of papers in each category is based on paper importance(based on our task) and then paper-release time.
>
> - [GPU comparison website](https://topcpu.net/gpu-c/GeForce-RTX-4090-vs-Tesla-V100-PCIe-32-GB)



## Old films/video restoration :fire:

> :dart: Current Working Direction!

- [ ] "DeOldify" open-sourced toolbox to restore image and video :+1:
  [code](https://github.com/jantic/DeOldify)

- [ ] "Bringing Old Photos Back to Life" CVPR oral, 2020 Apr :star:

  [paper(CVPR version)](https://arxiv.org/abs/2004.09484) [paper(TPAMI version)](https://arxiv.org/pdf/2009.07047v1.pdf) [code](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) [website](http://raywzy.com/Old_Photo/)

  [our Summary Note](./2020_CVPR_Bringing-Old-Photos-Back-to-Life_Note.md)

- [ ] "Bringing Old Films Back to Life" CVPR, 2022 Mar :star:

  [paper](https://arxiv.org/abs/2203.17276) [code](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life) [website](http://raywzy.com/Old_Film/)

  [our Summary Note](./2022_CVPR_Bringing-Old-Films-Back-to-Life_Note.md)

- [x] "DeepRemaster: Temporal Source-Reference Attention Networks for Comprehensive Video Enhancement" SIGGRAPH, 2019 Nov
  [paper](https://arxiv.org/abs/2009.08692) [website](http://iizuka.cs.tsukuba.ac.jp/projects/remastering/en/index.html)
  [our Note](./2019_SIGGRAPH_DeepRemaster-Temporal-Source-Reference-Attention-Networks-for-Comprehensive-Video-Enhancement_Note.md)

  > baseline in "Bringing Old Films Back to Life"  

- [ ] "DSTT-MARB: Multi-scale Attention Based Spatio-Temporal Transformers for Old Film Restoration" Master Thesis report, 2022 Sep
  [thesis report](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3023083/no.ntnu:inspera:118516831:64411262.pdf?sequence=1) [local pdf](./2022_master_report_DSTT-MARB--Multi-scale-Attention-Based-Spatio-Temporal-Transformers-for-Old-Film-Restoration.pdf)

  > 硕士论文，里面整合了上面 3 个文章

- [ ] "VRT: A Video Restoration Transformer" ArXiv, 2022 Jun
  [paper](https://arxiv.org/abs/2201.12288) [code](https://github.com/JingyunLiang/VRT?utm_source=catalyzex.com)

- [ ] "Recurrent Video Restoration Transformer with Guided Deformable Attention" NeurlPS, 2022 June
  [paper](https://arxiv.org/abs/2206.02146) [code](https://github.com/JingyunLiang/RVRT?utm_source=catalyzex.com)

  > VRT 的后续工作‘

- [x] "Blind Video Deflickering by Neural Filtering with a Flawed Atlas" CVPR, 2023 Mar :star:
  [paper](https://arxiv.org/abs/2303.08120) [code](https://github.com/ChenyangLEI/All-In-One-Deflicker?utm_source=catalyzex.com) [website](https://chenyanglei.github.io/deflicker/)
  [our note](./2023_05_CVPR_Blind-Video-Deflickering-by-Neural-Filtering-with-a-Flawed-Atlas_Note.md)

  > **用 Nerf 类似的 atlas 处理视频一致性问题**
  >
  > 有公布数据  <a name="Blind flickering Dataset"></a> 60 * old_movie, 大多为 350 帧图像; 21* old_cartoon, 大多为 50-100 帧;
  > 用 [RE:VISION. De:flicker](https://revisionfx.com/products/deflicker/) 去用软件人工修复（存在新手修的质量差的问题）

- [ ] "RTTLC: Video Colorization with Restored Transformer and Test-time Local" CVPR, 2023 Mar
  [paper](https://openaccess.thecvf.com/content/CVPR2023W/NTIRE/html/Li_RTTLC_Video_Colorization_With_Restored_Transformer_and_Test-Time_Local_Converter_CVPRW_2023_paper.html)

- [ ] "Exemplar-based Video Colorization with Long-term Spatiotemporal Dependency" Arxiv, 2023 Mar
  [paper](https://arxiv.org/abs/2303.15081)

- [ ] "BiSTNet: Semantic Image Prior Guided Bidirectional Temporal Feature Fusion for Deep Exemplar-based Video Colorization" ArXiv, 2022 Dec
  [paper](https://arxiv.org/abs/2212.02268) [website](https://yyang181.github.io/BiSTNet/?utm_source=catalyzex.com)

- [ ] "Modernizing Old Photos Using Multiple References via Photorealistic Style Transfer" CVPR 2023
  [paper](https://arxiv.org/abs/2304.04461) [code](https://github.com/KAIST-VICLab/old-photo-modernization) [website](https://kaist-viclab.github.io/old-photo-modernization/?utm_source=catalyzex.com)
  提供了老照片数据，可以下载

- [ ] "ReBotNet: Fast Real-time Video Enhancement" AeXiv, 2023 Mar
  [paper](https://arxiv.org/abs/2303.13504) [website](https://jeya-maria-jose.github.io/rebotnet-web/?utm_source=catalyzex.com)

- [ ] "SVCNet: Scribble-based Video Colorization Network with Temporal Aggregation" Arxiv, 2023 Mar
  [paper](https://arxiv.org/abs/2303.11591) [code](https://github.com/zhaoyuzhi/SVCNet)

- [ ] "Self-Prior Guided Pixel Adversarial Networks for Blind Image Inpainting" TAPMI, 2023 June
  [paper](https://ieeexplore.ieee.org/abstract/document/10147235)

- [ ] "SB-VQA: A Stack-Based Video Quality Assessment Framework for Video Enhancement" CVPR, 2023 May
  [paper](https://arxiv.org/abs/2305.08408)

  > mention `old film restoration`



## Diffusion in Video

> https://github.com/showlab/Awesome-Video-Diffusion :+1:
> https://scholar.google.com/scholar?as_ylo=2023&q=diffusion+video&hl=zh-CN&as_sdt=0,5

- [ ] "Diffusion Video Autoencoders: Toward Temporally Consistent Face Video Editing via Disentangled Video Encoding" CVPR oral, 2023 Dec, DVA
  [paper](https://arxiv.org/abs/2212.02802) [code](https://github.com/man805/Diffusion-Video-Autoencoders)
  [our note](./2023_CVPR_Diffusion-Video-Autoencoders--Toward-Temporally-Consistent-Face-Video-Editing-via-Disentangled-Video-Encoding_Note.md)

- [ ] "Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models" arxiv, 2023 Apr, **VideoLDM** :star:
  [paper](https://arxiv.org/abs/2304.08818) [website](https://research.nvidia.com/labs/toronto-ai/VideoLDM/) [code: unofficial implementation](https://github.com/srpkdyy/VideoLDM.git)
  [our note](./2023_CVPR_Align-your-Latents--High-Resolution-Video-Synthesis-with-Latent-Diffusion-Models_Note.md)

  > diffusion 用于 text2video，用预训练的 stable-diffusion，对 U-net 加 temporal layer 实现时序一致性

- [ ] "VideoComposer: Compositional Video Synthesis with Motion Controllability"
  [![Star](https://camo.githubusercontent.com/f3e411ac406a8793396b60a88e445f2b46ab95fc46d0d0376607ea93e1fac6b9/68747470733a2f2f696d672e736869656c64732e696f2f6769746875622f73746172732f64616d6f2d76696c61622f766964656f636f6d706f7365722e7376673f7374796c653d736f6369616c266c6162656c3d53746172)](https://github.com/damo-vilab/videocomposer) [![arXiv](https://camo.githubusercontent.com/0835af0e1376c6ea0c5c19fc50d0824d82eec71980e055575cb87b55a74f8b39/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f61725869762d6233316231622e737667)](https://arxiv.org/abs/2306.02018) [![Website](https://camo.githubusercontent.com/3e5ac86a01b8da4c1744c6b481736db4f759253d7b2bd1c6ee2bf1882146717f/68747470733a2f2f696d672e736869656c64732e696f2f62616467652f576562736974652d396366)](https://videocomposer.github.io/)
  [our note](./2023_06_VideoComposer--Compositional-Video-Synthesis-with-Motion-Controllability_Note.md)

  > Video LDM 上加入各种样式的 condition 实现可控视频生成

- [ ] [2023_Pix2Video: Video Editing using Image Diffusion >>paper with code](https://paperswithcode.com/paper/pix2video-video-editing-using-image-diffusion)

- [x] [2022_ICLR_DDNM_Zero-Shot-Image-Restoration-Using-Denoising-Diffusion-Null-Space-Model](./2022_ICLR_DDNM_Zero-Shot-Image-Restoration-Using-Denoising-Diffusion-Null-Space-Model_Note.md)

- [ ] [HistoryNet](https://github.com/BestiVictory/HistoryNet#historynet)

- [x] Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models

  https://arxiv.org/abs/2304.08818

- [x] VideoComposer

  https://github.com/damo-vilab/videocomposer

- [ ] Dreambooth

  https://github.com/zanilzanzan/DreamBooth

- [ ] LDMVFI: Video Frame Interpolation with Latent Diffusion Models
  https://arxiv.org/abs/2303.09508

- [ ] [Make-Your-Video: Customized Video Generation Using Textual and Structural Guidance](./2023_June_Make-Your-Video--Customized-Video-Generation-Using-Textual-and-Structural-Guidance.pdf)



## Editing

- [ ] "Layered Neural Atlases for Consistent Video Editing" SIGGRAPH, 2021 Sep
  [paper](https://arxiv.org/abs/2109.11418) [website](https://layered-neural-atlases.github.io/)
- [ ] "Stitch it in Time: GAN-Based Facial Editing of Real Videos" SIGGRAPH, 2019 Jan, STIT
  [paper](https://arxiv.org/abs/2201.08361) [code](https://github.com/rotemtzaban/STIT) [website](https://stitch-time.github.io/)
  [our note](./2022_SIGGRAPH_STIT_Stitch-it-in-Time--GAN-Based-Facial-Editing-of-Real-Videos_Note.md)



## Denoising

> [Awesome-Deblurring](https://github.com/subeeshvasu/Awesome-Deblurring)

- [ ] Learning Task-Oriented Flows to Mutually Guide Feature Alignment in Synthesized and Real Video Denoising
  [paper](https://arxiv.org/pdf/2208.11803v3.pdf)

- [ ] Recurrent Video Restoration Transformer with Guided Deformable Attention (RVRT, NeurlPS2022)

  https://paperswithcode.com/paper/recurrent-video-restoration-transformer-with Previous SOTA :moyai:

- [ ] Real-time Controllable Denoising for Image and Video
  https://paperswithcode.com/paper/real-time-controllable-denoising-for-image

- [x] "Low-Dose CT with a Residual Encoder-Decoder Convolutional Neural Network (RED-CNN)" 

  [paper](https://arxiv.org/abs/1702.00288) [code](https://github.com/SSinyu/RED-CNN/tree/master)
  [local pdf](./2017_TMI_RED-CNN_Low-Dose-CT-with-a-Residual-Encoder-Decoder-Convolutional-Neural-Network.pdf)

  > 医学 CT 去噪（噪声 GT 对），模型结构很简单

- [x] "Let Segment Anything Help Image Dehaze" Arxiv, 2023 Jun

  [paper](https://arxiv.org/abs/2306.15870)
  [our note](./2023_06_Let-Segment-Anything-Help-Image-Dehaze_Note.md)

  > 将 SAM 分割结果作为通道扩展到 U-net 模块中，进行去雾

 

## **Colorization**

> https://github.com/MarkMoHR/Awesome-Image-Colorization :star: 

- [ ] Interactive Deep Colorization

  https://github.com/junyanz/interactive-deep-colorization

- [ ] Improved Diffusion-based Image Colorization via Piggybacked Models  *Apr 2023*

  https://piggyback-color.github.io/

- [ ] Video Colorization with Pre-trained Text-to-Image Diffusion Models‘

  https://colordiffuser.github.io/

- [ ] "Deep Exemplar-based Video Colorization" CVPR, 2019 Jun
  [paper](https://arxiv.org/abs/1906.09909) [code](https://github.com/zhangmozhe/Deep-Exemplar-based-Video-Colorization)
  [local pdf](./2019_CVPR_Deep-Exemplar-based-Video-Colorization.pdf)



## Enhancement and Restoration :scissors:

- [ ] "Depth-Aware Video Frame Interpolation" CVPR, 2019 Apr, **DAIN**
  [paper](https://arxiv.org/abs/1904.00830) [code](https://github.com/baowenbo/DAIN)
- [ ] "ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks" ECCV, 2018 Sep, **ESRGAN(Enhanced SRGAN)**
  [paper](https://arxiv.org/abs/1809.00219) [code](https://github.com/xinntao/ESRGAN)
- [ ] "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data" ICCV, 2021 Aug
  [paper](https://arxiv.org/abs/2107.10833) [code](https://github.com/xinntao/Real-ESRGAN)
- [ ] BasicSR (**Basic** **S**uper **R**estoration) is an open-source **image and video restoration** toolbox
  [github repo](https://github.com/XPixelGroup/BasicSR)





## **Diffusion/GAN related**

- [ ] "A Style-Based Generator Architecture for Generative Adversarial Networks" CVPR, 2019 Dec, **StyleGAN**
  [paper](https://arxiv.org/abs/1812.04948) [code](https://nvlabs.github.io/stylegan2/versions.html)
  [our note](./StyleGAN_Note.md)

- [x] "Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model" ICLR Notable-Top-25%, 2022 Dec, **DDNM** :star:
  [paper](https://wyhuai.github.io/ddnm.io/) [website](https://wyhuai.github.io/ddnm.io/)
  [our note](./2022_ICLR_DDNM_Zero-Shot-Image-Restoration-Using-Denoising-Diffusion-Null-Space-Model_Note.md)

  > 将图像修复任务的数学模型，转换到 Range-Null space 分解，对于分解的其中一项替换为 Diffusion 的 noise 实现修复操作，融入 diffusion 的方式值得借鉴。

- [ ] "Palette: Image-to-Image Diffusion Models" SIGGRAPH, 2022 Nov
  [paper](https://arxiv.org/abs/2111.05826) [website](https://iterative-refinement.github.io/palette/) [code: unofficial implementation](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
  [our note](./2022_SIGGRAPH_Palette-Image-to-Image-Diffusion-Models_Note.md)

- [x] "Taming Transformers for High-Resolution Image Synthesis" CVPR, 2020 Dec, **VQ-GAN** :star:

  [paper](https://arxiv.org/abs/2012.09841) [website](https://compvis.github.io/taming-transformers/)
  [our note](./2021_CVPR_VQGAN_Taming-Transformers-for-High-Resolution-Image-Synthesis_Note.md)

- [ ] "High-Resolution Image Synthesis with Latent Diffusion Models" CVPR, 2022 Dec, **StableDiffusion**
  [paper](https://arxiv.org/abs/2112.10752) [github](https://github.com/CompVis/stable-diffusion) ![GitHub Repo stars](https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social)
  [our note](./2022_CVPR_High-Resolution Image Synthesis with Latent Diffusion Models_Note.md)

- [ ] [2022_Understanding Diffusion Models-A Unified Perspective.pdf](./2022_Understanding Diffusion Models-A Unified Perspective.pdf) :+1:

- [ ] DDPM

- [ ] DDIM

- [ ] IDDPM

- [ ] DDRM

- [ ] Score-based

- [ ] BeatGAN

  https://github.com/openai/guided-diffusion

- [ ] [VAE 博客](https://zhuanlan.zhihu.com/p/34998569) 提供了一个将概率图跟深度学习结合起来的一个非常棒的案例
  [code](https://github.com/bojone/vae)

- [ ] CycleGAN and pix2pix in PyTorch https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

- [x] Zero-Shot Image Restoration Using Denoising Diffusion Null-Space Model :+1:

  - [x] paper
  - [ ] code

- [ ] Palette: Image-to-Image Diffusion Models

- [ ] [2021_CVPR_StyleCLIP-Text-Driven-Manipulation-of-StyleGAN-Imagery.pdf](./2021_CVPR_StyleCLIP-Text-Driven-Manipulation-of-StyleGAN-Imagery.pdf)
  [Or Patashnik](https://orpatashnik.github.io/), [Zongze Wu ](https://www.cs.huji.ac.il/~wuzongze/), [Eli Shechtman ](https://research.adobe.com/person/eli-shechtman/), [Daniel Cohen-Or ](https://www.cs.tau.ac.il/~dcor/), [Dani Lischinski](https://www.cs.huji.ac.il/~danix/)

- [ ] [2020_CVPR_InterFaceGAN_Interpreting-the-Latent-Space-of-GANs-for-Semantic-Face-Editing.pdf](./2020_CVPR_InterFaceGAN_Interpreting-the-Latent-Space-of-GANs-for-Semantic-Face-Editing.pdf)

- [ ] [2022_WACV_Pik-Fix-Restoring-and-Colorizing-Old-Photos.pdf](./2022_WACV_Pik-Fix-Restoring-and-Colorizing-Old-Photos.pdf)

- [ ] [Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold](https://github.com/XingangPan/DragGAN) :moyai:

  - [ ] DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing :star:

- [ ] Pix2Video

- [ ] Self-Supervised Learning with Random-Projection Quantizer for Speech Recognition

  参考 random-projection 操作

- 小波变换

  - [ ] Wavelet Diffusion Models are fast and scalable Image Generators :+1:
  - [ ] [2023_Fourmer-An Efficient Global Modeling Paradigm for Image Restoration.pdf](./2023_Fourmer-An Efficient Global Modeling Paradigm for Image Restoration.pdf)

- Image2Image Translation

  - [ ] [2023_Zero-shot-Image-to-Image-Translation.pdf](./2023_Zero-shot-Image-to-Image-Translation.pdf)
    [github repo](https://github.com/pix2pixzero/pix2pix-zero)

- [ ] Locally Hierarchical Auto-Regressive Modeling for Image Generation :+1:

  https://github.com/kakaobrain/hqtransformer

- [ ] JPEG Artifact Correction using Denoising Diffusion Restoration Models

- [ ] Scalable Diffusion Models with Transformers

- [ ] All are Worth Words: A ViT Backbone for Diffusion Models

- [ ] Implicit Diffusion Models for Continuous Super-Resolution

- [ ] LayoutDM: Transformer-based Diffusion Model for Layout Generation

- [ ] Vector Quantized Diffusion Model for Text-to-Image Synthesis

- [ ] Image Super-Resolution via Iterative Refinement

- [ ] Real-World Denoising via Diffusion Model

- [ ] Diffusion in the Dark A Diffusion Model for Low-Light Text Recognition

- [ ] Exploiting Diffusion Prior for Real-World Image Super-Resolution

  https://iceclear.github.io/projects/stablesr/?utm_source=catalyzex.com
  [2023_preprint_Exploiting-Diffusion-Prior-for-Real-World-Image-Super-Resolution.pdf](./2023_preprint_Exploiting-Diffusion-Prior-for-Real-World-Image-Super-Resolution.pdf)

- [ ] Privacy Leakage of SIFT Features via Deep Generative Model based Image Reconstruction

- [ ] DreamDiffusion https://mp.weixin.qq.com/s/RDXINIvJvU_6FMoiX42bKg

- [ ] T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models
  [paper](https://arxiv.org/abs/2302.08453) [code](https://github.com/TencentARC/T2I-Adapter)
  可控生成

- [ ] SWAGAN: A Style-based Wavelet-driven Generative Model  23.7.14

- [ ] Shifted Diffusion for Text-to-image generation 23.7.15

- [ ] [Generative image inpainting with contextual attention](http://openaccess.thecvf.com/content_cvpr_2018/html/Yu_Generative_Image_Inpainting_CVPR_2018_paper.html) :star:

  类似 non-local attention  

- [ ] "A Unified Conditional Framework for Diffusion-based Image Restoration" 23.7.16
  [paper](https://arxiv.org/abs/2305.20049) [code](https://github.com/zhangyi-3/UCDIR) [website](https://zhangyi-3.github.io/project/UCDIR/)





## **Segmentation**

> [paper-list](https://github.com/liliu-avril/Awesome-Segment-Anything)

- [ ] SAM
  [paper](https://arxiv.org/abs/2304.02643) [code](https://github.com/facebookresearch/segment-anything)
  [Note](./2023_04_preprint_SAM_Segment-Anything_Note.md)

- [ ] Fast Segment Anything https://github.com/CASIA-IVA-Lab/FastSAM >> FPS25

- [x] UniAD CVPR23 Best paper

- [ ] MobileSAM https://mp.weixin.qq.com/s/zTakIRIsWOUBiUr3yn5rhQ

- [ ] Video SAM

  https://mp.weixin.qq.com/s/hyf_DnEdbUTh8VpeP0Mq-w

- [ ] [2023_June_Let-Segment-Anything-Help-Image-Dehaze.pdf](./2023_June_Let-Segment-Anything-Help-Image-Dehaze.pdf)
  https://arxiv.org/pdf/2306.15870.pdf

**Transformer  & Attention:moyai:**

- [ ] On the Expressivity Role of LayerNorm in Transformer's Attention  
  https://github.com/tech-srl/layer_norm_expressivity_role 这个工作可以加深对 transformer 的一些理解 :star:

- [ ] HaloNet padding 方式的注意力

  Scaling Local Self-Attention for Parameter Efficient Visual Backbones

- [ ] **SwimTransformer**

- [ ] SpectFormer: Frequency and Attention is what you need in a Vision Transformer  
  https://github.com/badripatro/SpectFormers

- [ ] Learning A Sparse Transformer Network for Effective Image Deraining 
  https://github.com/cschenxiang/DRSformer

- [ ] FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting :star: 时序数据
  https://github.com/MAZiqing/FEDformer

- [ ] Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification
  https://github.com/onuriel/PermutedAdaIN  这个工作很简单但是对于提升模型鲁棒性很有效，大家都可以看一下

  理解核心的那一小段代码即可

- [ ] [A ConvNet for the 2020s](https://arxiv.org/abs/2201.03545)



**MAE**

- [ ] **MAE**

- [ ] Siamese Masked Autoencoders 

  Masked Siamese Networks for Label-Efficient Learning
  https://github.com/facebookresearch/msn

  MixMask: Revisiting Masking Strategy for Siamese ConvNets
  https://github.com/LightnessOfBeing/MixMask
  这几个半监督/自监督的工作很有意思，大家好好看下

- [ ] SimMIM: a Simple Framework for Masked Image Modeling 

- [ ] Hard Patches Mining for Masked Image Modeling https://mp.weixin.qq.com/s/YJFDjcTqtX_hzy-FXt-F6w

- [ ] Masked-Siamese-Networks-for-Label-Efficient-Learning



**NLP & 多模态**

- [ ] Multimodal Prompting with Missing Modalities for Visual Recognition
  https://github.com/YiLunLee/Missing_aware_prompts  训练或者测试是多模态非完美情况
- [ ] Is GPT-4 a Good Data Analyst
  https://github.com/damo-nlp-sg/gpt4-as-dataanalyst

**对比学习**

- [ ] CCPL: Contrastive Coherence Preserving Loss for Versatile Style Transfer 
  https://github.com/JarrentWu1031/CCPL

## **HDR**

- [ ] "Inverting the Imaging Process by Learning an Implicit Camera Model" CVPR, 2023, Apr
  [paper](https://arxiv.org/abs/2304.12748) [website](https://xhuangcv.github.io/neucam/)
  [our note](./2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model_Note.md)

  > Represent the visual signal using implicit coordinate-based neural networks is recent trend in CV. Existing methods directly conider using the whole NN to represent the scene, and **not consider the camera separately.**
  > The paper **proposed a new implicit camera model (using implicit neural network)** to represent the physical imaging process. 使用 NeRF 单独模拟相机模型和 scene，实现解耦，增加可调节能力

- [ ] "BokehMe: When Neural Rendering Meets Classical Rendering" CVPR oral, 2022 Jun
  [paper](https://arxiv.org/abs/2206.12614v1) [website](https://juewenpeng.github.io/BokehMe/)

  > 对图像实现**可控的**模糊，调整焦距，光圈等效果。发布了数据集

- [ ] "DC2: Dual-Camera Defocus Control by Learning to Refocus" CVPR, 2023 Apr
  [website](https://defocus-control.github.io/)

  > **image refocus** requires deblurring and blurring different regions of the image at the same time, that means that image refocus is at least as hard as DoF Control

- [ ] DRHDR: A Dual branch Residual Network for Multi-Bracket High Dynamic Range Imaging

- [ ] Networks are Slacking Off: Understanding Generalization Problem in Image Deraining

- [ ] Image-based CLIP-Guided Essence Transfer

- [ ] Luminance Attentive Networks for HDR Image and Panorama Reconstruction
  https://github.com/LWT3437/LANet 

- [ ] Perceptual Attacks of No-Reference Image Quality Models with Human-in-the-Loop

- [ ] Perception-Oriented Single Image Super-Resolution using Optimal Objective Estimation 

- 小波变换 >> 无损

  - A simple but effective and efficient global modeling paradigm for image restoration
  - 周满



## NeRF

- [x] [2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model_Note.md](./2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model_Note.md)
- [ ] Neural Volume Super Resolution
  https://github.com/princeton-computational-imaging/Neural-Volume-Super-Resolution  NeRF+SR
- [x] LERF: Language Embedded Radiance Fields 
  https://github.com/kerrj/lerf  NeRF + 3D CLIP
- [ ] Implicit Neural Representations for Image Compression
  https://github.com/YannickStruempler/inr_based_compression 
- [ ] iNeRF: Inverting Neural Radiance Fields for Pose Estimation 
- [ ] ViP-NeRF: Visibility Prior for Sparse Input Neural Radiance Fields 
- [ ] AdaNeRF: Adaptive Sampling for Real-time Rendering of Neural Radiance Fields
- [ ] 2022_CVPR_Aug-NeRF--Training-Stronger-Neural-Radiance-Fields-with-Triple-Level-Physically-Grounded-Augmentations >> 输入增加扰动



## **Neural Operators**

- [ ] Factorized Fourier Neural Operators
  https://github.com/alasdairtran/fourierflow
- [ ] Super-Resolution Neural Operator
  https://github.com/2y7c3/Super-Resolution-Neural-Operator
- [ ] Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers
  https://github.com/NVlabs/AFNO-transformer
- [ ] Fourier Neural Operator for Parametric Partial Differential Equations
  https://github.com/neuraloperator/neuraloperator



## IQA

> :grey_question: what is IQA [CVPR IQA 博客](https://zhuanlan.zhihu.com/p/154017806)
> IQA(image quality assessment) Task target: quantification of human perception of image quality
>
> - Application
>   想对某一项视觉任务评估图像能否满足需要，比如针对人脸识别的质量评价，看一幅图像是否应该拒绝还是输入到人脸识别系统中；texture classification；texture retrieval （texture similarity）；texture recovery
> - 对于图像下游任务：denoising, deblurring, super-resolution, compression，能够提升图像质
> - Full Reference, No-reference 

- [x] "Image Quality Assessment: Unifying Structure and Texture Similarity" TPAMI, 2020 Dec, DISTS
  [paper](https://ieeexplore.ieee.org/abstract/document/9298952)
  [our note](./2020_TPAMI_DISTS_Image-Quality-Assessment-Unifying-Structure-and-Texture-Similarity_Note.md)

  > 针对有明显纹理的原图，让模型对 JPEG 压缩后、resample 的图像打分（实际上肉眼看上去 JPEG 更加模糊），之前方法对于 JPEG 图像质量评分错误地高于 resample 图。

- [x] "Re-IQA: Unsupervised Learning for Image Quality Assessment in the Wild" CVPR, 2023 Apr
  [paper]()
  [our noted pdf](./2023_CVPR_Re-IQA-Unsupervised-Learning-for-Image-Quality-Assessment-in-the-Wild.pdf)

  > 一种 NR-IQA 算法，使用对比学习的方式，使用 2 个 Res50 去学习 content & image-quality-aware features. 最后加一个 regressor 输出 image quality scores.
  > 对于 quality feature 一路，模仿 MoCoV2 ，修改了构造正负样本的方式进行训练。
  >
  > - `Full-reference IQA` 方法 对于 images in the wild 场景，没有 reference 应用受限
  >
  >   FR-IQA 需要参考图像（undistorted） & distorted 图像，一起才能输出评分。
  >
  > - high-level content representation using MoCoV2
  >
  >   2 crops from same image -> similar scores, but not the case for some human viewers.



# Research Note



## PaperWriting

> Latex Reference 直接在 Google research 上获取引用格式

- Abstract

  按句子：1 背景；2-3 存在问题；4-5-6 提出的方法，创新点；7实验证明有效

- introduction

  按段落可参考一下方式

  P1 介绍的任务 2-3C. 应用 & 但存在啥问题。早期的工作 3-4 个，**按总结的角度来列举**，不要只写 A propose B propose ...（挑3-4 个非常有名 or 典型的）

  P2 **按时间线**早期到近期写。但是还存在啥问题。

  P3 提出的方法

- related work 可以开 3 个小章节

  例如本文用到了 CLIP, MAE 在 LDR 任务上，可以分为以下三个小节介绍

  - LDR
  - CLIP
  - MAE

- Proposed Methods

  按一下顺序

  - overview

    按完整流程，结合公式详细，有条理的讲清楚本文的方法。同时，Framework 图下面 caption要很清楚，因为一般审稿人会先看 methods overview -> framework_graph。

  - 几个 proposed module xx

  - loss function

  > - 期刊：3.4 Methods 写完一般写到 6.5 页。初稿不算 reference 写12-13页，DL 实验部分一般4.5页
  >
  > - 公式
  >
  >   变量斜体，操作/运算符用正体 （Relu）





## ResearchSuggestion

- 论文笔记

  主要看懂方法；总结学到了啥（总结创新点，分析如何能用到自己的任务；提到的典型方法；...）

  **Key-point**

  **Contributions**

  **Related Work**

  **methods**

  **Limitations**

  **Summary :star2:**

  > learn what & how to apply to our task

  

- 文章阅读建议

  每周 5 篇精读，一篇文章要看多次的，第一次看完不懂没关系，但要记录下来后面再看！**一定要整理 code，进性总结 :star::star:** 

  > https://gaplab.cuhk.edu.cn/cvpapers/#home  这里整理分类了近几年计算机视觉方面重要会议（CVPR，ICCV，ECCV，NeurIPS，ICLR）的文章和代码，大家可以多看看
  >
  > https://openaccess.thecvf.com/menu 这是CVF的官网，一些计算机视觉一些重要会议（CVPR，ICCV，WACV）的所有文章附录等材料
  >
  > https://www.ecva.net/index.php 这是ECCV的官网，历年的文章附录都有
  >
  > 建议这些会议（CVPR，ICCV，ECCV，NeurIPS，ICLR，ICML，AAAI，IJCAI，ACMMM等）的文章以及一些重要期刊（T-PAMI，T-IP，TOG，TVCG，IJCV，T-MM，T-CSVT等）大家多阅读，相同或者相近任务的文章至少全部粗读一遍，然后选择性精读，需要学会使用Google学术和GitHub查询有用资料

1. 至少想 2 个创新点，做实验看为什么不 work，分析问题&看文献；

   > Possible direction

   - diffusion 稳定 or 加速训练

   - ControlNet >> 能否借鉴到 video editing :star:

   - GAN 之前存在的问题，一些思想能否用到 diffusion 中

     - 模式崩塌：多样性很差，只生产几个类别，质量比较好

     - **Limited Data 看思想**

     - wavelet diffusion models

   - Rado, 张雷老师组 >> diffusion model in low level

   - https://orpatashnik.github.io/ 看一下这个组的工作 >> StyleCLIP, StyleGAN-NADA
     [Daniel Cohen-Or Blog](https://danielcohenor.com/publications/)

2. 关注自己的研究方向，作为主线：**diffusion model 用于老电影修复**。
   当这周的论文阅读量没做完，**优先看自己的主线方向论文和项目进展**！

3. 主线方向，和视频相关方向都要看，只不过要学会某些进行略读。不要局限于技术细节，识别哪些可以暂时跳过，记录下来后面看。



# Current Progress :dart:

Main stream: Old video restoration!

## TODO :blue_book:

- [ ] 调研 old films 修复的 SOTA 论文	7.17
- [ ] 调研 old films 数据集	7.17



## Dataset

> focus on data source, data format

- FFHQ(Flickr-Faces-Hight-Quality)

  > [FFHQ 介绍博客](https://zhuanlan.zhihu.com/p/353168785)

  FFHQ是一个高质量的人脸数据集，包含1024x1024分辨率的70000张PNG格式高清人脸图像，在年龄、种族和图像背景上丰富多样且差异明显，在人脸属性上也拥有非常多的变化，拥有不同的年龄、性别、种族、肤色、表情、脸型、发型、人脸姿态等，包括普通眼镜、太阳镜、帽子、发饰及围巾等多种人脸周边配件，因此该数据集也是可以用于开发一些人脸属性分类或者人脸语义分割模型的。

- "Blind flickering" paper 提供构造的 flickering 数据  （<a href="#Blind flickering Dataset">Link to paper info</a>）

  - 60 * old_movie，存储为 `%05d.jpg` 大多为 350 帧图像，若 fps=25，约为 10-14s的视频。
  - 21* old_cartoon，图像格式存储，大多为 50-100 帧，约为 1 - 4s 视频




### **Old photos Dataset**

> [老照片修复中心](http://www.lzpxf.com/portal.php)
> https://www.ancientfaces.com/photo/george-roberts/1328388
> [old photos textures](https://www.google.com/search?q=old+photo+texture&tbm=isch&ved=2ahUKEwiNn_vEoLbsAhUM5hoKHXBiCUwQ2-cCegQIABAA&oq=old+photo+texture&gs_lcp=CgNpbWcQAzICCAAyAggAMgIIADICCAAyAggAMgIIADICCAAyAggAMgIIAFC3GFiBJmCsLWgAcAB4AIABkgGIAagGkgEDMS42mAEAoAEBqgELZ3dzLXdpei1pbWfAAQE&sclient=img&ei=QhKIX432HIzMa_DEpeAE&bih=875&biw=1920&client=ubuntu&hs=gm2)



### Old Films Dataset

> old movie, old cartoon 都可以！
>
> [10downloader tools to Download Youtube Video](https://10downloader.com/en/74)
> [How Old Movies Are Professionally Restored | Movies Insider](https://www.youtube.com/watch?v=6NzfJJny3Yg)
> [中国电影修复馆](https://www.cfa.org.cn/cfa/gz/xf/index.html)
> [Baidu 智感超清服务](https://zhuanlan.zhihu.com/p/302719798)

- Commercial Old films 
  https://www.britishpathe.com/ 老电影商店 75英镑下载一个。。

- Youtube [Denis Shiryaev](https://www.youtube.com/@DenisShiryaev) Youtuber permit other to use the video for research in his video comment. 有给出 source video Youtube url

  [[4k, 60 fps] A Trip Through New York City in 1911](https://www.youtube.com/watch?v=hZ1OgQL9_Cw&t=12s) already restore by several algorithms :warning:
  [[4k, 60 fps] San Francisco, a Trip down Market Street, April 14, 1906](https://www.youtube.com/watch?v=VO_1AdYRGW8) >> tell what methods used to restore

- Youtube [guy jones](https://www.youtube.com/@guy_jones)
  [1902c - Crowd Ice Skating in Montreal (stabilized w/ added sound)](https://www.youtube.com/watch?v=_qjT1ToUx1Q)

- Youtube Old cartoon

  [A Day at the Zoo (1939)](https://www.youtube.com/watch?v=RtblQQvT2Nk&list=PL-F4vmhdMdiXIXZEDNQ3UFLXmQqjHguBA)

- 优酷搜索老电影

  [优酷 kux 文件转为 mp4](https://zhuanlan.zhihu.com/p/111764932)

- B 站博主

  https://www.bilibili.com/video/BV1dT411u7Hu/?spm_id_from=333.788&vd_source=eee3c3d2035e37c90bb007ff46c6e881
  https://www.bilibili.com/video/BV1oG41187Rp/?spm_id_from=333.999.0.0&vd_source=eee3c3d2035e37c90bb007ff46c6e881

  https://github.com/leiurayer/downkyi b站视频下载工具







## metrics :baby:

- PSNR

  `skimage\metrics\simple_metrics.py`

- SSIM

- LPIPS(Learned Perceptual Image Patch Similarity)

  CVPR2018的一篇论文《The Unreasonable Effectiveness of Deep Features as a Perceptual Metric》，更符合人类的感知情况。LPIPS的值越低表示两张图像越相似，反之，则差异越大。

- E_warp

- NIQE

- BRISQUE

- FID

  > [FID score for PyTorch](https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py)

  **分数越低代表两组图像越相似**，或者说二者的统计量越相似，FID 在最佳情况下的得分为 0.0，表示两组图像相同

  用 torchvision 中预训练好的 InceptionV3 模型（修改了几层），提取第几个 block的输出。对每个图像弄成 dimens 尺寸的 embedding。对这个 embedding 提取均值和方差

  ```python
  def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                      device='cpu', num_workers=1):
      """Calculation of the statistics used by the FID"""
      mu = np.mean(act, axis=0)
      sigma = np.cov(act, rowvar=False)
      return mu, sigma
  ```

  $ d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).$

  



## Diffusion basics

> [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)
> [知乎 Diffusion 博客](https://zhuanlan.zhihu.com/p/587727367)
>
> [survey github repo](https://github.com/chq1155/A-Survey-on-Generative-Diffusion-Model)
> [Diffusion Models in Vision: A Survey](https://ieeexplore.ieee.org/abstract/document/10081412)

### Forward diffusion process

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1- \beta_t} x_{t-1}, \beta_tI)
$$

均值 $\mu= \sqrt{1- \beta_t}，\sigma^2= \beta_t$ ，Let $\alpha_t = 1-\beta_t >> \mu = \sqrt{1 - \alpha_t}, \sigma=\sqrt{1-\alpha_t}$ .
根据 Reparameterization Trick，$z = \mu + \sigma \bigodot \epsilon$ ，得到 $x_t = \sqrt{1 - \alpha_t} + \sqrt{1-\alpha_t} * \epsilon_{t-1}, ~where~\epsilon\sim\mathcal{N}(0,I)$ :star:
$$
x_t = \sqrt{\overline{\alpha}} \cdot x_0 + \sqrt{1- \overline{\alpha}} \cdot \epsilon_0 \\
= \sqrt{\alpha_0 \ldots\alpha_{t-(T-1)}} x_{t-T} + \sqrt{1-(\alpha_0 \ldots\alpha_{t-(T-1)})} \epsilon_{t-T}  ~~where~timestamp=T
$$


#### Reparameterization Trick

> [博客参考](https://lilianweng.github.io/posts/2018-08-12-vae/#reparameterization-trick)
>
> - isotropic Gaussian distribution
>
>   isotropic (having uniform physical properties in all directions) 各同向性，**isotropic distribution 在不同方向上的分布都很均匀**，这应该是”各向同性“的直观解释。
>
>   ![isotropic_definition_graph.webp](./docs/isotropic_definition_graph.webp)
>
>   `isotropic random vector` 指的就是每个方向上方差为 1，不同方向上协方差为 0 的随机向量。**VAE的reparameterization那一层其实就是isotropic random vector**
>   $$
>   Definition:~\text{A random vector X in } R^n \text{is called isotropic if} \\
>   Cov(X) = E(XX^T) = I_n
>   $$
>
>   1. 判定X是否为 isotropic
>      isotropic 的随机向量与任意向量的内积都等于那个向量的 L2-norm
>   2. 高维的 isotropic random vector 几乎正交（向量内积==0）

Sampling is a stochastic process and therefore we cannot backpropagate the gradient. To make it trainable, the reparameterization trick is introduced. 
$$
z \sim q_{\phi}(z | x) = \mathcal{N}(z;\mu^{(i)},\sigma^{2(i)}I)\\
z = \mu + \sigma \bigodot \epsilon , where ~\epsilon\sim N(0,I)
$$
In the multivariate Gaussian case, we make the model trainable by **learning the mean and variance of the distribution, $\mu$ and $\sigma$,** explicitly using the reparameterization trick, while the stochasticity remains in the random variable $\epsilon\sim N(0,I)$. 

> 随机采样，转化为 $z = \mu + \sigma \bigodot \epsilon$ 形式，能够去学习 $\mu, \sigma$



### [Reverse diffusion process](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#parameterization-of-l_t-for-training-loss)

reverse the above process and sample from $q(x_{t-1} | x_t)$， we will be able to recreate the true sample from a Gaussian noise input $x_T \sim \mathcal{N}(0,1)$. Unfortunately, we **cannot easily estimate $q(x_{t-1} | x_t)$ because it needs to use the entire dataset** 

Thus we need to learn a neural network to approximate the conditioned probability distributions in the reverse diffusion process, $q(x_{t-1} | x_t) = \mathcal{N}(x_t; \mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))$. 

**We would like to train $\mu_{\theta}$ to predict $\tilde{\mu_t} = \frac{1}{\sqrt{\alpha_t}}\cdot(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\cdot\epsilon_t)$** :star:

- model predict $\epsilon_{t}$ from input $x_t$ at timestep=t
  $$
  \mu_{\theta} = \frac{1}{\sqrt{\alpha_t}}\cdot(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\cdot \epsilon_{\theta}(x_t,t))
  $$
  

  真实值根据 forward Diffusion process 的公式逆推得到 $\tilde{\mu_t} = \frac{1}{\sqrt{\alpha_t}}\cdot(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\cdot\epsilon_t)$

使用 $L_t$ （细节后续补）优化模型，使得预测的 $\mu_{\theta}$ 与 $\tilde{\mu_t}$ 差别最小





- 小结

  1. Forward diffusion process

     对原图 x0 **逐步（Timestep t）**施加一个**少量（通过 $\beta_i$ 来控制）**的高斯噪声，使得原图逐渐失去特征，得到 noise map
     $$
     q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1- \beta_t} x_{t-1}, \beta_tI) \\
     x_t = \sqrt{1 - \alpha_t} + \sqrt{1-\alpha_t} * \epsilon_{t-1}, ~where~\epsilon\sim\mathcal{N}(0,I)
     $$

  2. Reverse diffusion process

     希望根据  $q(x_{t-1} | x_t) = \mathcal{N}(x_t; \mu_{\theta}(x_t,t), \Sigma_{\theta}(x_t,t))$ 采样来逆转上述加噪声过程，得到原图。模型预测 $\mu_{\theta}$ 

  3. 加速 Diffusion Process

     - DDIM samples only a subset of S diffusion steps

     - LDM(*Latent diffusion model*) runs the diffusion process in the latent space instead of pixel space, making training cost lower and inference speed faster. 




## DDPM :baby:

> [2020_NeurIPS_DDPM_Denoising-diffusion-probabilistic-models.pdf](./2020_NeurIPS_DDPM_Denoising-diffusion-probabilistic-models.pdf)
> [博客参考](https://zhuanlan.zhihu.com/p/563661713)
> [Markov Chain 博客参考](https://zhuanlan.zhihu.com/p/274775796)
> [DDPM B站视频](https://www.bilibili.com/video/BV1b541197HX/?spm_id_from=333.337.search-card.all.click&vd_source=eee3c3d2035e37c90bb007ff46c6e881)  [DDPM 数学基础](https://zhuanlan.zhihu.com/p/530602852)

![DDPM_reverse_diffusion_understanding.png](./docs/DDPM_reverse_diffusion_understanding.png)



reverse diffusion 迭代一步
$$
q(x_{t-1}| x_t,x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}(x_t, x_0), \tilde{\beta_t}I)\\
$$
网络训练 $ \tilde{\mu}(x_t, x_0), \tilde{\beta_t}$ 从而构成生成模型，模型给出的均值方差通过 reparameteration trick $y_{t-1} = \mu + \sigma * \epsilon$ 

> $\epsilon \in \mathcal{N}(0,1)$ 随机采样得到？





## DDIM :baby:

> [keras implementation](https://keras.io/examples/generative/ddim/):+1:
> [2020_ICLR_DDIM_Denoising-Diffusion-Implicit-Models.pdf](./2020_ICLR_DDIM_Denoising-Diffusion-Implicit-Models.pdf)

$$
\sigma^2_t = \eta * \tilde{\beta_t} \\

\tilde{\beta_t} = \sigma^2_t = \eta *\beta_t  ~~ where~\eta~is~constant \\
$$

 when $\eta ==0$ 





# package

## diffusers

> [**Hugging Face** documentation](https://huggingface.co/docs/diffusers/main/en/tutorials/tutorial_overview)

