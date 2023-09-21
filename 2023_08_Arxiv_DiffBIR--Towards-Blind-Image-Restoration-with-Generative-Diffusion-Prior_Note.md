# DiffBIR

> "DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior" Arxiv, 2023 Aug
> [paper](https://arxiv.org/abs/2308.15070) [code](https://github.com/xpixelgroup/diffbir) [website](https://0x3f3f3f3fun.github.io/projects/diffbir/)
> [paper local pdf](./2023_08_Arxiv_DiffBIR--Towards-Blind-Image-Restoration-with-Generative-Diffusion-Prior.pdf)

## **Key-point**

设计了一个 unified framework 实现 blind image Restoration，分析了 BSR,ZIR,BFR 的局限性。

1. expanded degradation model

2. utilizes the well-trained **Stable Diffusion as the prior** >> 训练时 fix

   - injective modulation sub-network – `LAControlNet`

3. introduces a **two-stage** solution pipeline to ensure both realness and fidelity

   apply a Restoration Module (i.e., SwinIR) to reduce most degradations, and then finetune the Generation Module



## **Contributions**

## **Related Work**

- BSR(Blind SR)

  - "Designing a practical degradation model for deep blind image super-resolution" ICCV, 2021Mar
    [paper](https://arxiv.org/abs/2103.14006)
  - " Real-esrgan: Training real-world blind super-resolution with pure synthetic data"
    [paper](https://arxiv.org/abs/2107.10833) [code](https://github.com/xinntao/Real-ESRGAN)

  adversarial loss >> learning the reconstruction process in an end-to-end manner

- ZIR(zero-shot IR)

  预先假定了退化

- BFR(Blind face) >> sub-domain of BIR

  - CodeFormer :statue_of_liberty:

  - VQFR

  - DifFace

    "DifFace: Blind Face Restoration with Diffused Error Contraction"
    [code](https://github.com/zsyOAOA/DifFace)

    > 可以做 old photo enhancement

  **smaller image space, these methods can utilize VQGAN** and Transformer to achieve surprisingly good results



## **methods**

<img src="./docs/DiffBIR_2stage_pipeline.png" style="zoom:50%;" />

2 stage model

- Restoration Module

  > "SwinIR: Image Restoration Using Swin Transformer" ICCV, 2021 Aug
  > [paper](https://arxiv.org/abs/2108.10257)

  pretrain a SwinIR [36] on large-scale dataset to achieve the preliminary degradation removal across diversified degradations

-  the generative prior



## **Experiment**

> ablation study 看那个模块有效，总结一下





## **Limitations**

## **Summary :star2:**

> learn what & how to apply to our task