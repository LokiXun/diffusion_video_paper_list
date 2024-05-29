# U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation

> "U-Mamba: Enhancing Long-range Dependency for Biomedical Image Segmentation" Arxiv, 2024 Jan
> [paper](http://arxiv.org/abs/2401.04722v1) [code](https://github.com/bowang-lab/U-Mamba) [website](https://wanglab.ai/u-mamba.html) [pdf](./2024_01_Arxiv_U-Mamba--Enhancing-Long-range-Dependency-for-Biomedical-Image-Segmentation.pdf) [note](./2024_01_Arxiv_U-Mamba--Enhancing-Long-range-Dependency-for-Biomedical-Image-Segmentation_Note.md)
> Authors: Jun Ma, Feifei Li, Bo Wang

## Key-point

- Task
- Problems
- :label: Label:

## Contributions

## Introduction

## methods

## Code

使用 nnUNetv2 库，`self.network = self.build_network_architecture(...)` 实例化模型  [code](https://github.com/bowang-lab/U-Mamba/blob/548f3b217f279c22afc19f88afc5ad1c3895e932/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L205)

训练 forward 在 [`def run_training`](https://github.com/bowang-lab/U-Mamba/blob/548f3b217f279c22afc19f88afc5ad1c3895e932/umamba/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py#L1249) 中；

> [需要下载模型配置信息](https://drive.google.com/file/d/1v_l58gAAVCua2FXCoEVIWupLb_SlrwRV/view?usp=drive_link) `3d_fullres`







## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

