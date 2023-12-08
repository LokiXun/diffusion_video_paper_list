# VideoCrafter1: Open Diffusion Models for High-Quality Video Generation

> "VideoCrafter1: Open Diffusion Models for High-Quality Video Generation" Arxiv, 2023 Oct
> [paper](http://arxiv.org/abs/2310.19512v1) [code](https://ailab-cvc.github.io/videocrafter)  :+1:
> [pdf](./2023_10_Arxiv_VideoCrafter1--Open-Diffusion-Models-for-High-Quality-Video-Generation.pdf)
> Authors: Haoxin Chen, Menghan Xia, Yingqing He, Yong Zhang, Xiaodong Cun, Shaoshu Yang, Jinbo Xing, Yaofang Liu, Qifeng Chen, Xintao Wang, Chao Weng, Ying Shan

## Key-point

- Task: text-to-video (T2V) and image-to-video (I2V) tasks
- Background
- :label: Label:

������ Video Diffusion Milestone

Image-to-Video (I2V) model is designed to produce videos that strictly adhere to the content of the provided reference image

first open-source generic I2V foundation model that can preserve the content and structure of the given reference image.



## Contributions

## Related Work

- ������Ƶ���ɽ��

  "Benchmarking and Evaluating Large Video Generation Models"
  [website](https://evalcrafter.github.io/)

concatenating frame-wise depth map with input noise sequences for video editing

### Condition

- "VideoComposer: Compositional Video Synthesis with Motion Controllability"
  [paper](https://arxiv.org/abs/2306.02018)

- DragNUWA

  introduce trajectory control



## methods

![image-20231103151545684](docs/2023_10_Arxiv_VideoCrafter1--Open-Diffusion-Models-for-High-Quality-Video-Generation_Note/image-condition-branch.png)





## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

�Աȿ�Դ Text2Video ģ�ͣ���ȥ���Ľ��

![image-20231103152828991](docs/2023_10_Arxiv_VideoCrafter1--Open-Diffusion-Models-for-High-Quality-Video-Generation_Note/image-20231103152828991.png)

1. I2VGen-XL Ϊ image2Video, ����� SD-XL �ȸ����ı�����һ��ͼ
2. Pika Labs: best text-alignment performance but does not always generate the correct style
3. Gen-2 ��ʱ��������Ƶ���ı�ƥ��Ȳ���

> Pika [6] and Gen-2 [1] are well-known text2video generation products developed by commercial

![image-20231103154234272](docs/2023_10_Arxiv_VideoCrafter1--Open-Diffusion-Models-for-High-Quality-Video-Generation_Note/Image2Video_compare.png)

- ���ķ������ɵĵ�һ֡�����ã������ڿ�Դģ���У�һ���Ժͱ�����ۺ�����
- our I2V model still has several limitations such as the successful rate, unsatisfactory facial artifacts, etc, requiring further efforts for im



## Limitations

## Summary :star2:

> learn what & how to apply to our task

