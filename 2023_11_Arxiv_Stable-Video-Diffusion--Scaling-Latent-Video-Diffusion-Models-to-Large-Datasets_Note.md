# Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets

> "Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets" Arxiv, 2023 Nov
> [paper](http://arxiv.org/abs/2311.15127v1) [code](https://github.com/Stability-AI/generative-models) 
> [pdf](./2023_11_Arxiv_Stable-Video-Diffusion--Scaling-Latent-Video-Diffusion-Models-to-Large-Datasets.pdf)
> Authors: Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, Robin Rombach

## Key-point

- Task
- Problems
- :label: Label:



we dub Large Video Dataset (LVD), consists of 580M annotated video clip pairs, forming 212 years of content.



## Contributions

## Introduction

### CFG

![image-20240115205443091](docs/2023_11_Arxiv_Stable-Video-Diffusion--Scaling-Latent-Video-Diffusion-Models-to-Large-Datasets_Note/image-20240115205443091.png)



## methods

> 模型结构看 Appendix D

- Stage I: image pretraining, i.e. a 2D text-to-image diffusion model [13, 64, 71]. 

  先不加 temporal layer 在图像数据上训练

  - VideoLDM
  - "Imagen video: High definition video generation with diffusion models"
  - Make-A-Video: Text-to-Video Generation without Text-Video Data

- Stage II: video pretraining, which trains on large amounts of videos. 

  插入 temporal layer （VideoLDM 中的） 在 14 帧 256x384 提出的数据集 LVD 上训练

- Stage III: video finetuning, which refines the model on a small subset of high-quality videos at **higher resolution**

  Next, we finetune the model to generate 14 320 × 576 frames for 100k iterations using **batch size 768.**



> - it is crucial to adopt the noise schedule when training image diffusion models 
>   "simple diffusion: End-to-end diffusion for high resolution images"
>
> - In contrast to works that only train temporal layers [9, 32] or are completely training-free [52, 114], **we finetune the full model**
>
>   参考 VideoLDM "Align your Latents" 的 temporal layer



## Experiment

> ablation study 看那个模块有效，总结一下

- setting

  train all our models for 12k steps (∼16 hours) with 8 80GB A100 GPUs using a total batch size of 16, with a learning rate of 1e-5.



### img2video

 found that standard vanilla **classifier-free guidance [36] can lead to artifacts**: too little guidance may result in inconsistency with the conditioning frame while too much guidance can result in oversaturation

found it helpful to linearly increase the guidance scale across the frame axis (from small to high). Details can be found in App. D

## Limitations

## Summary :star2:

> learn what & how to apply to our task

