# DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior

> "DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior" Arxiv, 2023 Oct
> [paper](http://arxiv.org/abs/2310.16818v2) [code](https://github.com/deepseek-ai/DreamCraft3D) 
> [pdf](./2023_10_Arxiv_DreamCraft3D--Hierarchical-3D-Generation-with-Bootstrapped-Diffusion-Prior.pdf)
> Authors: Jingxiang Sun, Bo Zhang, Ruizhi Shao, Lizhen Wang, Wen Liu, Zhenda Xie, Yebin Liu

## Key-point

- Task: hierarchical 3D content generation
- Background
- :label: Label:��

1. ѵ��һ�� 3D diffusion model Ԥ�����ӽ�ͼ�񣬸������ɽ�Ϊ�ֲڵ�3D�ṹ
2. ���� Stable Diffusion ϸ��mesh ����ϸ��

> To sculpt geometries that render coherently, we perform **score distillation sampling via a view-dependent diffusion model**. We train a personalized diffusion model, Dreambooth, on the augmented renderings of the scene, imbuing it with 3D knowledge of the scene being optimized

- ��ע

  1. ���ʹ�� diffusion ����������ͬʱ���ı� diffusion ������

     ʹ�������ͼ��

  2. diffusion ѵ����μ��٣�

     �տ�ʼѵ�� timesteps ���� 70% > ֮�� 20%



## Contributions

## Related Work

- SDS Loss
  score distillation sampling (SDS) loss encourages the rendered images to match the distribution modeled by the diffusion model.
  $$
  \nabla_\theta\mathcal{L}_{\mathrm{SDS}}(\phi,g(\theta))=\mathbb{E}_{t,\boldsymbol{\epsilon}}\Big[\omega(t)(\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t;y,t)-\boldsymbol{\epsilon})\frac{\partial\boldsymbol{x}}{\partial\theta}\Big],
  $$
  

- classifierfree guidance (CFG) ��ʽ���� condition

  ʹ�� SDS Loss + CFG ����ģ��

- variational score distillation (VSD) loss

  



## methods

![image-20231101112443915](docs/2023_10_CVPR_DreamCraft3D--Hierarchical-3D-Generation-with-Bootstrapped-Diffusion-Prior_Note/DreamCraft3D_structure.png)

leverages a 2D image generated from the text prompt and uses it to guide the stages of geometry sculpting and texture boosting



### Diffusion timestep annealing

- utilize a view-conditioned diffusion model, Zero-1-to-3
  The Zero-1-to-3 is a fine-tuned 2D diffusion model, which **hallucinates the image** in a relative camera pose c given the reference image x. ʹ�� Diffusion ������ӽǵ�ͼ��

- Diffusion timestep annealing

  At the start of optimization, prioritize sampling larger diffusion timestep t from the range [0.7, 0.85] when computing Equation 6 to provide the global structure. As training proceeds, we linearly anneal the t sampling range to [0.2, 0.5] over hundreds of iterations



ѵһ���ϳ����ӽǵ� diffusion ģ�ͣ��������� 3D �ṹ����ѵ���� diffusion �ϳ����ӽ�ͼ��ϸ�ں�ƽ��ģ����leaves the texture blurry��



###  TEXTURE BOOSTING 

augment the texture realism, we use variational score distillation (VSD) loss,
$$
\nabla_\theta\mathcal{L}_\mathrm{VSD}(\phi,g(\theta))=\mathbb{E}_{t,\boldsymbol{\epsilon}}\left[\omega(t)(\boldsymbol{\epsilon}_\phi(\boldsymbol{x}_t;y,t)-\boldsymbol{\epsilon}_\mathrm{lora}(\boldsymbol{x}_t;y,t,c))\frac{\partial\boldsymbol{x}}{\partial\theta}\right]
$$
ʹ�� Stable Diffusion ϸ������3D mesh��



## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

## Limitations

## Summary :star2:

> learn what & how to apply to our task

