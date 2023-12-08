# Pix2Video: Video Editing using Image Diffusion

> "Pix2Video: Video Editing using Image Diffusion" ICCV, 2023 Mar
> [paper](http://arxiv.org/abs/2303.12688v1) [code](https://github.com/duyguceylan/pix2video) [website](https://duyguceylan.github.io/pix2video.github.io/)
> [pdf](./2023_03_Arxiv_Pix2Video--Video-Editing-using-Image-Diffusion.pdf)
> Authors: Duygu Ceylan, Chun-Hao Paul Huang, Niloy J. Mitra

## Key-point

- Task: Video Editing
- Problems
  1. ��α��ֱ༭�����Ƶ������ԭʼ��Ƶ�����ݣ��༭����Ƶ��һ���ԣ�
  2. ��֡�༭����֡��һ��
- :label: Label: `Training-Free`



## Contributions



## Related Work

Temporal Consistency: ����һ���༭���֡����ȡ�����ںϵ���ǰ֡�� self-attn layer��guided diffusion strategy

ͼ�� diffusion ģ��û����ʽ�ش����˶���Ϣ���ο� ControlNet ʹ�� depths or �ָ�ͼ�ṩ�˶���Ϣ

- patch-based video based stylization
  "Stylizing video by example"

- concurrent video editing

  "Tune-A-Video: One-shot tuning of image diffusion models for text-to-video generation" ��Ҫ finetune

- neural layered representations
  "Text2Live: Text-driven layered image and video editing" >> compute-intensive preprocessing



### Diffusion

> [survey](https://arxiv.org/abs/2209.04747)



## methods

����ʹ�ö������Ϣ������ depth ���ṩ�㹻���˶���Ϣ����������֡Ԥ�� depth ��Ϊ�������룻ͨ�� Self-attention Feature Injection ģ�����༭��һ�������⣬`Guided Latent Update` �����˸����

![image-20231124130318950](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124130318950.png)



### Self-attention Feature Injection

Stable Diffusion ʹ�� UNet �ṹ�� decoder����ʹ�� cross-attention layer ȥ��ע�������� cross-attention layer �޸ģ���ǰ֡��Ϊ Q���ο�֡����һ֡&��һ֡ concatenate����Ϊ KV

> ��������
>
> 1.  perform the above feature injection in the decoder layers of the UNet find effective in maintaining appearance consistency :star:
>
> 2. Decoder ���������ʾ�߷ֱ��ʣ����������ֻ�к�С�Ľṹ����
>
>    deeper layers of the decoder capture high resolution and appearance-related information and already result in generated frames with similar appearance but small structural changes
>
> 3. �� Encoder �����ֻ��������
>
>    do not observe further significant benefit when injecting features in the encoder of the UNet and observe slight artifacts

��˱����� decoder ��ʼ���� layer ���༭�������޸ĺ���첻�Ǻܴ������





### Guided Latent Update

- Motivation

  �������༭��ʽ�Ѿ�����һ���Ա༭����������˸������

Diffusion �� $x_{t} \to x_{t-1}$ ��ҪԤ�� $\hat{x_0}$���ٵõ� $x_{t-1}$
$$
\hat{x}_0^t=\frac{x_t-\sqrt{1-\alpha_t}\epsilon_\theta^t(x_t)}{\sqrt{\alpha_t}}
$$
����� x0 ʹ��ǰһ֡ x0 ����Լ�������������� L2 Loss �Ե�ǰ���� $x_{t-1}$ ���и��£���ֹǰ��֡����̫�󣬲ο�α���� Line19
$$
\begin{aligned}g(\hat{x}_0^{i,t},\hat{x}_0^{i-1,t})=\Vert\hat{x}_0^{i,t}-\hat{x}_0^{i-1,t}\Vert_2^2\end{aligned}
$$
һ�������� 50 ����ȥ�룬ֻ�Կ�ʼ�� 25�������� loss �� $x_t$ ���е������Ժ�������Ҳ���������ή��ͼ������

> "Diffusion models already have a semantic latent space." **���� overall structure of the generated image is already determined in the earlier diffusion steps**

���ҳ�ʼ **noise Ҳ����˸��Ӱ��**��ʹ�� Image Captions ģ�Ͷ���Ƶ��֡��ȡ prompt���Ӷ����� inversion ��ȡͼ������



Pix2Video �㷨����α����

![image-20231124134357172](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124134357172.png)







## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

**ʵ������**

- evaluate:  DAVIS dataset�����ķ�������Ҫѵ����

  length of these videos ranges from 50 to 82 frames

- ģ�ͽṹ

  depth-conditioned Stable Diffusion [1] as in our backbone diffusion model.

  Ԥ�ȱ���� video ÿ֡������

- metrics :star:

  capture the faithfulness **ʹ�� CLIP-Score �������ɽ����ʵ��**���༭�ı� & �༭���ͼ�� embedding �����������ƶ�

  "Imagen video: High definition video generation with diffusion models", "Tune-A-Video" 

  temporal coherency ����������֡���� average CLIP similarity  && ʹ�ù��� warp �������֡���� PixelMSE

  

- �Աȷ���

  "Stylizing video by example" ref21

  "Text2Live" ��Ҫ��Ԥ����õ�һ�� neural atlas Ҫ��ʱ 7-8 h��֮����ʱ��Ҫ finetune text2image ���ɣ���Ҫ 30min

  "SDEdit"

  "Tune-a-Video"



**���ķ�������Ч��**

Pix2Video ����ʹ�õ��� or ���ǰ�����壬�ܽ��оֲ� �����壩or ȫ�ֱ༭�����

![image-20231124140047949](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124140047949.png)



### �� SOTA ����ָ��Ƚ�

CLIP-Text , Image �����༭ͼ����༭Ŀ���һ���ԣ�PixelMSE ����ʱ��һ���ԣ�ʱ��һ���ԣ���˸����� "Text2Live" ��һЩ����������죬Text2Live ��Ҫ 7 h��ȥ���� atlas

![image-20231124141041782](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124141041782.png)



**�Ա�����Ч��**

![image-20231124141927961](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124141927961.png)

1. ref21 ʹ�ù�����Ϊ������Ϣ��������Ƶ���������壬�༭Ч�����ͺܶ�

   fails as new content becomes visible

2. Tex2Live ���ڶ��ǰ������ʧЧ

3. Tune-a-Video �༭ǰ����ṹһ���Բ���

4. SDEdit ��֡����ʱ��һ���Ժܲ�



### Ablations

attention layer ʹ�òο�֡���Ա�ʹ�� anchor ��һ֡��ǰһ֡��anchor+ǰһ֡��anchor+���ȡǰ��һ֡

Attending only to the previous frame or a randomly selected previous frame results in **temporal and structural artifacts**

������֡��Without an anchor frame, we observe more temporal flickering and the edit diminishes

![image-20231124142630281](docs/2023_03_ICCV_Pix2Video--Video-Editing-using-Image-Diffusion_Note/image-20231124142630281.png)



## Limitations

1. ���ɵ����ж�����䣬���Ŷ��Կ��Ż�
1. ʱ��һ���Խϲ��˸������Ȼ����
1. Ŀǰ��� 80 ֡ & ch�Ķ���Ƶ





## Summary :star2:

> learn what & how to apply to our task

1. 
