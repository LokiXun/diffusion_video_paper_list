# Palette :statue_of_liberty:

> [official blog](https://iterative-refinement.github.io/palette/) 
> [github repo](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)
> [2022_SIGGRAPH_Palette-Image-to-Image-Diffusion-Models.pdf](./2022_SIGGRAPH_Palette-Image-to-Image-Diffusion-Models.pdf)
> [**Colorization Task Benchmarks**](https://paperswithcode.com/task/colorization) [paper with code](https://paperswithcode.com/paper/palette-image-to-image-diffusion-models-1)
>
> - inpaint 用另一张不同角度 GT 补全？
> - 素描 -> 风格



**Background**

a simple and general framework for **image-to-image translation called Palette**. We've evaluated Palette on four challenging computer vision tasks, namely *colorization, inpainting, uncropping, and JPEG restoration*. Palette is able outperform strong task-specific GANs without any task-specific customization or hyper-parameter tuning.

<video src="https://iterative-refinement.github.io/palette/videos/palette_movie_v2.m4v" ></video>



**Contributions**

**Related Work**

- [ ] Pix2Pix: [2017_CVPR_Pix2Pix_Image-to-Image-Translation-with-Conditional-Adversarial-Networks.pdf](./2017_CVPR_Pix2Pix_Image-to-Image-Translation-with-Conditional-Adversarial-Networks.pdf) :moyai:

- Diffusion

  - [ ] [Diffusion Models Beat GANS on Image Synthesis](http://arxiv.org/abs/2105.05233). :moyai:

    https://github.com/openai/guided-diffusion?utm_source=catalyzex.com

  - [ ] [Wavegrad: Estimating gradients for waveform generation](https://wavegrad.github.io/) :rocket:
    [2021_ICLR_Wavegrad--Estimating-gradients-for-waveform-generatio.pdf](./2021_ICLR_Wavegrad--Estimating-gradients-for-waveform-generatio.pdf)  Conditional Diffusion

  - [ ] [D2c: Diffusion-decoding models for few-shot conditional generation](https://proceedings.neurips.cc/paper/2021/hash/682e0e796084e163c5ca053dd8573b0c-Abstract.html)

  - [ ] SR3 [Image super-resolution via iterative refinement](https://ieeexplore.ieee.org/abstract/document/9887996/)



## **methods**



### Evaluation Protocol

- Inception Score(IS)
- FID
- Classification Accuracy of pretrained Res50
- Perceptual Distance(PD)





**Unsolved Limitations**





 **Summary**

> learn what & how to apply to our task