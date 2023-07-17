# Implicit camera model

> [2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model.pdf](./2023_CVPR_Inverting-the-Imaging-Process-by-Learning-an-Implicit-Camera-Model.pdf)
> [official blog](https://xhuangcv.github.io/neucam/) :star:
>
> 算力：一个 V100
>
> - What is HDR or LDR image?](https://www.easypano.com/showkb_228.html)
>   **dynamic range** refers to *ratio between the brightest and darkest*（最亮与最暗亮度的比值） parts of the scene. **The Dynamic Range of real-world scenes can be quite high - ratios of 100,000:1** are common in the natural world.
>
>   - HDR(High Dynamic Range)
>
>     HDR image (image of 32bit/color channel) shows the dynamic range of real world (natural dynamic range is **generally considered to be 100,000:1 （十万比一）**
>
>     the **dynamic range human eyes** can identify is around 100,000:1
>
>     So from broad sense （consider the standard display equipment）, image with **dynamic range of higher than 255:1 (8 bit per cooler channel)** is regarded as HDR image.
>
>   - LDR(Low Dynamic Range)
>
>     Dynamic range of JPEG format image won't exceed 255:1, so it is considered as LDR (Low Dynamic Range).
>
>   - [Tone mapping 进化论](https://zhuanlan.zhihu.com/p/21983679)
>
>     对于 HDR Image 如果要显示到 LDR 设备，**需要 tone mapping 的过程，把HDR变成LDR**
>
>   
>
> - camera response function
>   $$
>   P = f(E) \\
>   \text{where E is the image's irradiance, and P is the pixels' value}
>   $$
>
> - ghost artifact
>
>   > [***What is ghosting and how do you measure it?*** ](https://mriquestions.com/ghosting.html)
>
>   ***Ghosting*** is a type of structured noise appearing as repeated versions of the main object (or parts thereof) in the image. They occur because of signal instability between pulse cycle repetitions. Ghosts are usually blurred, smeared, and shifted and are most commonly seen along the phase encode direction. One of the most famous of these is the so-called ***Nyquist N/2 ghost***,
>
> - image intensity 表示每个通道的图像的像素灰度值（值的大小）
>
> - [radiance & irradiance](https://blog.csdn.net/u010476094/article/details/44106203)
>
>   irradiance 入射光线
>
> - [White balance](https://www.cambridgeincolour.com/tutorials/white-balance.htm)
>
>   White balance (WB) is the **process of removing unrealistic color casts, so that objects which appear white in person are rendered white in your photo**. Proper camera white balance has to take into account the "color temperature" of a light source, which refers to the relative warmth or coolness of white light. Our eyes are very good at judging what is white under different light sources, but digital cameras often have great difficulty with auto white balance (AWB)
>
> 

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

