## STIT :star:

<video src="https://user-images.githubusercontent.com/24721699/153860260-a431379e-ebab-4777-844d-4900a448cf85.mp4" ></video>

> [2022_SIGGRAPH_STIT_Stitch-it-in-Time--GAN-Based-Facial-Editing-of-Real-Videos.pdf](./2022_SIGGRAPH_STIT_Stitch-it-in-Time--GAN-Based-Facial-Editing-of-Real-Videos.pdf)
> [official blog & code](https://stitch-time.github.io/) >> 看效果
> [blogs Note](https://www.casualganpapers.com/hiqh_quality_video_editing_stylegan_inversion/Stitch-It-In-Time-explained.html)
>
> - Datasets
>   - FFHQ >> pretrained StyleGAN
> - GPU configuration >> 一块 2080 



**Background**

**StyleGAN and several insightful tweaks** to the to the frame-by-frame inversion and editing pipeline to obtain a method that produces temporally consistent high quality edited videos, and yes, that includes CLIP-guided editing

**Contributions**

- stitching editing methods 优化 edited image 的边界
- e4e+ PTI 缓解 PTI distortion-editability trade-off
- 2 metrics: TL-ID, TG-ID

**Related Work**

- PTI
- e4e



**Limitations & Summary**

- The biggest one is speed. Unfortunately, the method does not work in realtime, which would be perfect for real world applications. Alas, a single 5 sec video takes 1.5 hours per edit
- Hair sometimes gets cropped incorrectly 
- lose details after edited, little bit cartoon-like



**methods**

> 1st author's group

![STIT_pipeline.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\STIT_pipeline.jpg)



### Alignment :airplane:

The pretrained StyleGAN are trained on **FFHQ datasets whose data are preprocessed, so we need similar crop-align preprocessing**. But cropping are sensitive to facial landmark. :question:

A **gaussian lowpass filter** is applied to detected landmarks before cropping and aligning faces from video frames to reduce sensitivity to the exact locations of facial landmarks

> 预训练的 StyleGAN 在预先处理过的 FFHQ数据上得到，为了一致，要把自己的视频模仿 FFHQ 数据，crop + align 人脸区域
>
> - gaussian lowpass filter >> [2021_StyleVideoGAN-A-Temporal-Generative-Model-using-a-Pretrained-StyleGAN.pdf](./2021_StyleVideoGAN-A-Temporal-Generative-Model-using-a-Pretrained-StyleGAN.pdf)
> - [python face_alignment blog](https://blog.csdn.net/qq_41185868/article/details/103219220) :+1:



#### StyleVideoGAN

> [2021_StyleVideoGAN-A-Temporal-Generative-Model-using-a-Pretrained-StyleGAN.pdf](./2021_StyleVideoGAN-A-Temporal-Generative-Model-using-a-Pretrained-StyleGAN.pdf)

Alignment using lowpass gaussian filter.



### Inversion

StyleGAN's latent space W suffers distortion-editability tradeoff (the editted part is inconsistent with origin).

**Using PTI to solve this distortion-editability tradeoff.** PTI finds more-editable pivot latent code and finetune the generator that produce better-reconstructing latent code. 

> PTI 用来增强 latent code 的重建和修改能力

- :question: Why us e4e as encoder?

  1. PTI 得到的 pivots 之间很远，editing 后不匹配
  2. 如果 PTI 需要修改某些属性（FFHQ 数据每张图有对应的属性特征 eg:beard），只有几帧的 pivots latent code 的 attr 需要修改，会造成只改那几帧引起不连贯。

  **e4e 学习 lower frequency representation, 作者希望几帧之间连贯的变化，对应到 latent code 的变化也连贯**

![STIT_inversion_PTI_loss.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\STIT_inversion_PTI_loss.jpg)



#### PTI

> [blogs](https://www.casualganpapers.com/sota-fine-tuning-stylegan-inversion/PTI.html)

#### e4e



### Editing

> :question: 如何按方向修改

using off-the-shelf linear editing techniques in StyleCLIP & InterFaceGAN.

得到 senmantic direction $\delta{w}$, 使用 PTI generator，获取 edite_face  $e_i = G(w_i + \delta{w}; \theta_{PTI})$ 



#### StyleCLIP :star:

> - [2021_CVPR_StyleCLIP-Text-Driven-Manipulation-of-StyleGAN-Imagery.pdf](./2021_CVPR_StyleCLIP-Text-Driven-Manipulation-of-StyleGAN-Imagery.pdf)
>   [Or Patashnik](https://orpatashnik.github.io/), [Zongze Wu ](https://www.cs.huji.ac.il/~wuzongze/), [Eli Shechtman ](https://research.adobe.com/person/eli-shechtman/), [Daniel Cohen-Or ](https://www.cs.tau.ac.il/~dcor/), [Dani Lischinski](https://www.cs.huji.ac.il/~danix/)

#### InterFaceGAN

> [2020_CVPR_InterFaceGAN_Interpreting-the-Latent-Space-of-GANs-for-Semantic-Face-Editing.pdf](./2020_CVPR_InterFaceGAN_Interpreting-the-Latent-Space-of-GANs-for-Semantic-Face-Editing.pdf)



- Naive Editing

  ```shell
  input_folder="/home/ps/Desktop/dataset/frames_clip_test/curry/"
  output_folder="./edit_result/curry/"
  run_name="curry_test"
  edit_name="gender"
  # StyleCLIP
  python edit_video.py --input_folder ${input_folder} \
          --output_folder ${output_folder} \
          --run_name ${run_name} \
          --edit_name ${edit_name} \
          --edit_range -8 -8 1 \
          --edit_type "styleclip_global" \  # interfacegan
          --neutral_class="female face" \  # 想要改成的样子
          --target_class="face" \  # 描述原始视频中的位置
          --beta=0.2  # 阈值
  
  # InterfaceGAN
  python edit_video.py --input_folder ${input_folder} \
          --output_folder ${output_folder} \
          --run_name ${run_name} \
          --edit_name ${edit_name} \
          --edit_range -8 -8 1 \
          --edit_type "interfacegan" \  
  ```

  

  





### Stitching Tuning

![STIT_stitching_tuning_module.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\STIT_stitching_tuning_module.jpg)

a second tuning is performed because Simply pasting the edited face back into the original frame leads to undesirable artifacts such as ghostly outlines around the border.

1. use `BiSeNet2` to obtain a **segmentation mask** for the edited face using a pretrained segmentation network, **dilate the mask** to create the boundary region
2. paste the edited face into the original frame and finetune the generator for each frame to make the boundary region more similar to the original image without changing the edited face.



#### latent-Transformer :star:

> [2021_ICCV_A-Latent-Transformer-for-Disentangled-Face-Editing-in-Images-and-Videos.pdf](./2021_ICCV_A-Latent-Transformer-for-Disentangled-Face-Editing-in-Images-and-Videos.pdf)

STIT  中使用该论文中的 seamless cloning 方法，将修改过的图接到原图背景上，用作对比实验。
We use **Poisson image editing method** to blend the modified faces with the original input frames. In order to blend only the face area, we use the segmentation mask obtained from the detected facial landmarks.

- Poisson blending

  [博客参考](https://zhuanlan.zhihu.com/p/453095752)

  ![Poisson-image-editing-method_effect.jpg](C:\Users\Loki\workspace\LearningJourney_Notes\Tongji_CV_group\docs\Poisson-image-editing-method_effect.jpg)



#### BiSeNetV2

> [github repo](https://github.com/CoinCheung/BiSeNet)

- `class ContextPath(nn.Module)`







### proposed metrics

> [2022_ArcFace-Additive-Angular-Margin-Loss-for-DeepFace-Recognition.pdf](./FaceFeature/2022_ArcFace-Additive-Angular-Margin-Loss-for-DeepFace-Recognition.pdf)

using ArcFacce to evaluate the identity similarity score of each pari of frames.

- TL(Local)-ID

  adjacent frames >> average whole video

- TG(global)-ID

  not adjacent 



### summary

> **discuss learn what & how to apply to our task here!**

- Learn the general pipline of face video editing
- StyleGAN, PTI, E4E, Latent-transformer Previous GAN-related video-editing methods
- metrics
- how to design the loss to achieve target & show generated result in images as experiment result



