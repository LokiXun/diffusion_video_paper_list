# DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion

> "DreamPose: Fashion Image-to-Video Synthesis via Stable Diffusion" Arxiv, 2023 Apr
> [paper](http://arxiv.org/abs/2304.06025v4) [code](https://github.com/johannakarras/DreamPose) [website](https://grail.cs.washington.edu/projects/dreampose/)
> [pdf](./2023_04_Arxiv_DreamPose--Fashion-Image-to-Video-Synthesis-via-Stable-Diffusion.pdf)
> Authors: Johanna Karras, Aleksander Holynski, Ting-Chun Wang, Ira Kemelmacher-Shlizerman

## Key-point

- Task: generating animated fashion videos from still images.
- Background
- :label: Label:

transform a **pretrained text-to-image model (Stable Diffusion [16])** into a pose-and-image guided video synthesis model, using a novel finetuning strategy

 a set of architectural changes to support the **added conditioning signals, and techniques to encourage temporal consistency**



## Contributions

## Related Work

### Conditioning Mechanisms for Diffusion Models

- "Instructpix2pix: Learning to follow image editing instructions" CVPR, 2022 Nov :star:
  [paper](https://arxiv.org/abs/2211.09800)

  ͼ��༭���񣨸ķ�񣬸����壩 image conditioning signals are often **concatenated with the input noise** to the denoising U-Net �ʺ�**spatially aligned**�����



## methods

modify the original Stable Diffusion architecture in order to enable image and pose conditioning.  replace the CLIP text encoder with a dual CLIP-VAE image encoder and adapter module![image-20231103194019558](docs/2023_04_Arxiv_DreamPose--Fashion-Image-to-Video-Synthesis-via-Stable-Diffusion_Note/DreamPose-fremwork.png) 

1. ���ݼ�����һ��С��Χ��ȥ fine-tune a pretrained Stable Diffusion model����Ҫ accept additional conditioning signals (image and pose), 
2. At inference time, we generate each frame independently && ��֡  with uniformly distributed Gaussian nois

����a method to effectively **condition the output image on target pose** while also enabling temporal consistency

InstructPix2Pix ֱ�ӽ�ͼ�� noise concat���ʺϿռ�ֲ�һ�µ�������˷���������һ����̬������ռ䲼���в������Բ��ʺ��ô˷�����

- finetune Stable Diffusion ���� Prior Loss

  > �ο� ControlNet�� InstructPix2Pix

  A crucial objective �������ԭ�� SD ѵ�����ݽӽ���

  �Ķ���U-net input ����ն���� Pose��concat �� noise ���棩��Cross-Attn ����context��CLIP image feature + VAE ��������΢�� VAE Decoder

- Image Condition

  1. ������ InstructPix2Pix ����ȥ condition���ʺ���������ռ䲼��һ�µ����
     �� Pose ���ɵ�֡������ͼ�ڿռ䲼�����в���û����
  2. ����뷨���� CLIP image embedding �滻 text-embedding�������� in practice that **CLIP image embeddings alone are insufficient** for capturing fine-grained details in the conditioning image��CLIP ͼ������ϸ�ڲ���

  ������CLIP ͼ��embedding + VAE ���������⣺architecture does not support VAE latents as a conditioning signal by default����� add an adapter module A that combines the CLIP and VAE embeddings

  һ��ʼѵ����� Adapter ģ�� ControlNet���� VAE ��Ȩ����Ϊ0��ֻ�� CLIP ����

- Pose Condition: �� InstructPix2Pix ��ʽ condition

  ������֡�غ�noise �ӣ�����ͼ���֡�����ܴ��ü�֡ pose ������һ��� noise ����concatenate the noisy latents $z_i$ with a target pose representation $c_p$ ��5�� pose��������

  modify the UNet ���� layer �� 10 ��channel��5�� pose ������������ģ�ͽṹ

  > image-conditional diffusion methods
  > "Dreambooth: Fine tuning text-to-image diffusion models for subject-driven"
  > "Dreamfusion: Text-to-3d using 2d diffusion"
  > "Dreamix: Video diffusion models are general video editors"
  >
  > Pose ͼ���ֳɵ� DensePose Ԥ��

  1. ����ֻѵһ֡ >> texture-sticking���� DA Random Crop
  2. finetuning the VAE decoder

- ƽ��ͼ�� condition �� Pose condition ��ǿ��
  modulate the strength of image conditioning cI and pose conditioning cp during inference using dual **classifier-free guidance **

  > �ο� InstructPix2Pix



## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

- setting

  UBC Fashion dataset�� 339 train & 100 test videos. Each video has a frame rate of 30 frames/second and is approximately 12 seconds long��15w֡��ƽ��ÿ����Ƶ 360 ֡��

- ���� Stable Diffusion Image Conditioning ������ʵ��

  ![image-20231103203624467](docs/2023_04_Arxiv_DreamPose--Fashion-Image-to-Video-Synthesis-via-Stable-Diffusion_Note/image-20231103203624467.png)

- �Ƿ� Finetune VAE

  ![image-20231103202506283](docs/2023_04_Arxiv_DreamPose--Fashion-Image-to-Video-Synthesis-via-Stable-Diffusion_Note/image-20231103202506283.png)





## Limitations

1. ���ɵ�һ���Բ��㣬֡��������˸����

2. finetuning and inference times are slow com

   a specific subject takes approximately 10 minutes for the UNet and 20 minutes for the VAE decoder

   **18 second per-frame rendering time.**



## Summary :star2:

> learn what & how to apply to our task

