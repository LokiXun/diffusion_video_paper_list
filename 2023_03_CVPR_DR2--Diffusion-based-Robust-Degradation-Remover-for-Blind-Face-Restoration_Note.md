# DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration

> "DR2: Diffusion-based Robust Degradation Remover for Blind Face Restoration" CVPR, 2023 Mar :star:
> [paper](https://arxiv.org/abs/2303.06885) [code](https://github.com/Kaldwin0106/DR2_Drgradation_Remover)
> [paper local pdf](./2023_03_CVPR_DR2--Diffusion-based-Robust-Degradation-Remover-for-Blind-Face-Restoration.pdf)

## Key-point

> Blind face restoration 任务

将图像中的高低频信息分开处理（分别从不同来源获取），然后再将二者融合后的信息以condition方式插入diffusion模型的复原过程中，促进diffusion复原过程中的生成能力，再通过一个增强模型进行后处理

- degradation remover模块，移除了图像中产生的退化
- 增强模型，对diffusion生成的、平滑的结果进行进一步的增强，本文中使用VQFR实现

**Contributions**

## Related Work

exploitation of various facial priors





## methods

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task