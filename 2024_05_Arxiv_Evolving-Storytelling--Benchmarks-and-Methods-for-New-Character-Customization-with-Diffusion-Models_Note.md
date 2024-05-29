# Evolving Storytelling: Benchmarks and Methods for New Character Customization with Diffusion Models

> "Evolving Storytelling: Benchmarks and Methods for New Character Customization with Diffusion Models" Arxiv, 2024 May 20
> [paper](http://arxiv.org/abs/2405.11852v1) [code]() [pdf](./2024_05_Arxiv_Evolving-Storytelling--Benchmarks-and-Methods-for-New-Character-Customization-with-Diffusion-Models.pdf) [note](./2024_05_Arxiv_Evolving-Storytelling--Benchmarks-and-Methods-for-New-Character-Customization-with-Diffusion-Models_Note.md)
> Authors: Xiyu Wang, Yufei Wang, Satoshi Tsutsui, Weisi Lin, Bihan Wen, Alex C. Kot

## Key-point

- Task: story visualization

- Problems

  - generating content-coherent images for storytelling tasks

  - 出现极少的特殊人物，生成较难，训练集里面没有先验

    > However, it falls short when generating stories featuring unseen new characters like Slaghoople because it has little prior knowledge of her

    

- :label: Label:



 Our customized model can take only one story of Slaghoople and generate new stories involving both new and existing characters



## Contributions

- introduce NewEpisode benchmark(dataset), refined text prompts and eliminates character leakage

> introduce the NewEpisode benchmark, comprising refined datasets designed to evaluate generative models’ adaptability in generating new stories with fresh characters using just a single example story

- propose EpicEvo
  - introduce adversarial character alignment module to **align the generated with exemplar images**

> we propose EpicEvo, a method that . customizes a diffusion-based visual story generation model with a single story featuring the new characters seamlessly integrating them into established character dynamics.



## Introduction

- the absence of a suitable benchmark due to potential character leakage and inconsistent text labeling

- the challenge of distinguishing between new and old characters, leading to ambiguous results



- Q：character leakage?



## methods

## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

