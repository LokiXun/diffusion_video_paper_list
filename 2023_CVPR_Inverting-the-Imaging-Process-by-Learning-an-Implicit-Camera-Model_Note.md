# Implicit camera model

> [2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model.pdf](./2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model.pdf)
> [official blog](https://xhuangcv.github.io/neucam/) :star:
>
> 算力：一个 V100

Represent the visual signal using implicit coordinate-based neural networks is recent trend in CV. Existing methods directly conider using the whole NN to represent the scene, and **not consider the camera separately.**
The paper **proposed a new implicit camera model** to represent the physical imaging process.

- two image inverse task 展示 implicit representation 的效果

  1. all-in-focus image: multi-focus images fusing
  2. HDR imaging: increase the dynamic range from multi-LDR images

- paper proposed

  devise a blur-generator and implicit tone-mapper to model the aperture and exposure of the camera's imaging process.

**introduction & related work**

- little literature trys to find the implicit representation for camera model separately
- existing NeRF methods assume the RGB value is captured precisely but not actually the case
  1. ISP may alter the image's luminance change, DoF
  2. physical issue: light rays get through aperture may blur the image
- restore the scene's raw content by inversing from the images is challenge

**Implicit Neural Representation** (**NeRF methods drawback**)

1. only model camera's forward mapping；only build tone-mapper module with NeRF
2. **ignore the camera model**
   is an unified coordinate-based NeRF(conisder scene and camera in one NN), **not sure whether the implicit camera model apply the to other camera settings**, like current images may taken from cameras with auto-focus and different camera params.
3. Hard to control defocus-blur and exposure simultaneously
4. not able to process dynamic scenes



**HDR image**

1. ghost artifacts

   有些用 optical-flow 去解决，但是需要额外数据

2. 适用 learning-based HDR methods, 需要监督训练，数据不足



**Multi-Focus Image Fusion**

1. GAN methods to fuse multi-focus images

   需要大量训练数据，all-in-focus images 数据不多

2. 很多输入的 multi-focus images 未对齐

3. 无法处理动态的 multi-focus images



**methods**

1. implicit camera model
2. self-supervised framework for image-enhancement
3. experiment to show the application result

