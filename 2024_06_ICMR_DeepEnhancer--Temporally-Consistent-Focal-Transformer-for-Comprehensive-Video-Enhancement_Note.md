# DeepEnhancer: Temporally Consistent Focal Transformer for Comprehensive Video Enhancement

> "DeepEnhancer: Temporally Consistent Focal Transformer for Comprehensive Video Enhancement" ICMR, 2024 Jun 7
> [paper](https://dl.acm.org/doi/pdf/10.1145/3652583.3658031) [code](https://github.com/jiangqin567/DeepEnhancer) [pdf](./2024_06_ICMR_DeepEnhancer--Temporally-Consistent-Focal-Transformer-for-Comprehensive-Video-Enhancement.pdf) [note](./2024_06_ICMR_DeepEnhancer--Temporally-Consistent-Focal-Transformer-for-Comprehensive-Video-Enhancement_Note.md)
> Authors: Qin Jiang, Qinglin Wang, Lihua Chi, Wentao Ma, Feng Li, Jie Liu

## Key-point

- Task
- Problems
- :label: Label:

## Contributions

## Introduction

## methods

## setting

- .WeimplementourmodelsusingthePyTorch frameworkandperformthetrainingonfourNVIDIAV100GPUs



### data

- 训练 & 测试数据

> Datasets.Referringtotherelatedworks[21,41],weadopt REDS[31],DAVIS[33],Videvo[1]datasetsandareal-worldold filmdatasetdownloadfromtheInternetfortrainingandtesting.
>
> - "Learning blind video temporal consistency" ECCV
> - "Bringing old films back to life" CVPR

合成数据

>
> TheREDSisalargedatasetwithsignificantmotionanddiverse
>
>  The DAVIS datasetisdesignedforvideosegmentation,whichcon tains60trainingvideosand30testingvideos.
>
> TheVidevodataset, containing20videos, isselectedforadditional testing.



真实老电影

> Thereal worldoldfilmdatasetsSet1andSet2aregatheredfortesting,which include13and12oldmoviesrespectively.



只展示了合成数据



### metric

- PSNR, SSIM, LPIPS, FID
- NIQE(NaturalnessImageQualityEvaluator)
- CDC(ColorDistributionConsistency)





## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what

### how to apply to our task

