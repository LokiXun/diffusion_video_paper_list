# High-Resolution Image Synthesis with Latent Diffusion Models (stable diffusion)

> [github](https://github.com/CompVis/stable-diffusion) ![GitHub Repo stars](https://img.shields.io/github/stars/CompVis/stable-diffusion?style=social)
> [pdf](./2022_CVPR_High-Resolution Image Synthesis with Latent Diffusion Models.pdf)
> [Youtube 博主解读](https://www.youtube.com/watch?v=f6PtJKdey8E) [知乎博客](https://zhuanlan.zhihu.com/p/597247221) [towardScience 博客](https://towardsdatascience.com/paper-explained-high-resolution-image-synthesis-with-latent-diffusion-models-f372f7636d42) :+1:
> [What can Stable Diffusion do?](https://stable-diffusion-art.com/how-stable-diffusion-work/)
> [Stable Diffusion with 🧨 Diffusers](https://huggingface.co/blog/stable_diffusion#how-does-stable-diffusion-work)



## **KeyPoint**

**Contributions**

- 大幅降低训练和采样阶段的计算复杂度，让文图生成等任务能够在消费级GPU上，在10秒级别时间生成图片，大大降低了落地门槛



## **Related Work**

image generation has been tackled mainly through **four families of models**: Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs), Autoregressive Models (ARMs), Diffusion Probabilistic Models(DMs)

- GANs: show promising results for **data with limited variability**, bearing mode-collapse, **unstable training**

  > **Mode collapse**: *this phenomenon occurs when the generator can alternately generate a limited number of outputs that fool the discriminator. In general, GANs **struggle capturing the full data distribution.***

- VAE(Variational Autoencoders)

  not suffer from mode-collapse and can efficiently generate high-resolution images. **sample quality** is **not** always **comparable to** that of **GANs**.

- DPM

  operation in **pixel space** by adding or removing noise to a tensor of the same size as the original image results in **slow inference speed** and **high computational cost**

## **methods**

> - :question:  VAE?
> - KL-reg, VQ-reg

`High-Resolution Image Synthesis with Latent Diffusion Models`, we can break it down into **four main steps** :star:

### Perceptual Image Compression

**Latent Diffusion explicitly separates the image compression** phase to **remove high frequency details (perceptual compression**), and learns a semantic and conceptual composition of the data (**semantic compression**).

> 原有的非感知压缩的扩散模型有一个很大的问题在于，由于在像素空间上训练模型，如果我们希望生成一张分辨率很高的图片，这就意味着我们训练的空间也是一个很高维的空间。
> 引入感知压缩就是说**通过 VAE 这类自编码模型对原图片进行处理，忽略掉图片中的高频信息**，只保留重要、基础的一些特征。
>
> 感知压缩主要利用一个预训练的自编码模型，该模型能够学习到一个在**感知上等同于图像空间的潜在表示空间 latent space**。优势是只需要训练一个通用的自编码模型，就可以用于不同的扩散模型的训练，在不同的任务上（image2image，text2image）使用。
>
> **基于感知压缩的扩散模型的训练**本质上是一个两阶段训练的过程，第一阶段需要训练一个自编码器，第二阶段才需要训练扩散模型本身。
>
> $\mathcal{x} \in R^{H\times W \times 3} \rightarrow R^{h\times w \times 3} $，下采样因子 $f = H/h = W/w = 2^m$ 
>
> - 压缩比的选取，比较 FID，Inception Scores vs. training progress



#### Optimization

> Details in Appendix G

We train all our autoencoder models in an adversarial manner following VQGAN

![VQGAN_approach_overview.jpg](./docs\VQGAN_approach_overview.jpg)

- Regularization

  - KL(Kullback-Leibler) 散度 [博客参考](https://zhuanlan.zhihu.com/p/100676922)

    **计算两个概率分布之间差异**，当用一个去近似另一个分布，计算会有多少损失。ps: KL 散度很像距离，但不是，因为 :Loss_KL(p|q) != Loss_KL(q|p) 不对称！





### Latent Diffusion Models

LDM 在训练时就可以利用编码器得到 $z_t$
$$
L_{DM} := E_{\epsilon(x),\varepsilon\sim\mathcal{N}(0,1),t}{[\abs{\abs{ \epsilon - \epsilon_\theta(x_t,t)}}^2_2]}
\\
L_{LDM} := E_{\epsilon(x),\varepsilon\sim\mathcal{N}(0,1),t}{[\abs{\abs{ \epsilon - \epsilon_\theta(z_t,t)}}^2_2]}
$$


### Conditioning Mechanisms





**Limitations**



**Summary**

> learn what & how to apply to our task

# Code

> pytorch-lighting  >> save moves for zero the gradient,...
> [stable-diffusion code analysis blog](https://zhuanlan.zhihu.com/p/613337342)
>
> - Summary
>
>   有了 x 和 condition info ( class_name )
>
>   1. 将 x 通过 VQGAN encoder 映射为 latent code
>
>   2. DDPM 训练
>
>      1. 从 0-1000 随机取 timestep 
>
>      2. condition  经过 `ClassEmbedder `映射得到 1x512 tensor
>
>      3. `def p_losses` 从干净 x0 加噪 T 步
>
>         随机取噪声 noise $\epsilon$ (shape 和 x0 一样)
>
>         `def q_sample` 按公式加噪 $q(x_t | x_0)\sim \mathcal{N}(\sqrt{\bar{a_t}} x_0, (1-\bar{a_t})I)$
>
>      4. 调 U-net 输入 x_t, t, condition 预测噪声，与之前随机取的 noise $\epsilon$ 计算 L2 loss

```
# training config
configs/autoencoder/autoencoder_kl_64x64x3.yaml  # dataset image->256x256x3
configs/latent-diffusion/cin-ldm-vq-f8.yaml
```

- Loading module from yaml info >> use same code to initialize different classes

  ```python
  import importlib
  
  """ yaml example
      unet_config:
        target: ldm.modules.diffusionmodules.openaimodel.UNetModel
        params:
          image_size: 32
  """
  
  def instantiate_from_config(config):
      if not "target" in config:
          if config == '__is_first_stage__':
              return None
          elif config == "__is_unconditional__":
              return None
          raise KeyError("Expected key `target` to instantiate.")
      return get_obj_from_str(config["target"])(**config.get("params", dict()))
  
  
  def get_obj_from_str(string, reload=False):
      module, cls = string.rsplit(".", 1)
      if reload:
          module_imp = importlib.import_module(module)
          importlib.reload(module_imp)
      return getattr(importlib.import_module(module, package=None), cls)
  ```

  

## AutoencoderKL

`def training_step` main training procedures happens

> VQGAN code learning

- **LPIPS Loss >> perceptual loss**

  input & reconstruction from Autoencoder all shapes like [B, 3, 256, 256]

  use **pretrained `VGG16`'s feature Module**, which is `nn.Sequential` hence the layer could be visited by index. What `LPIPS` do is **split the starting 30 layers into 5 slice `nn.Sequential`** (cut at layer indexing by 3,8,15,22,29 and **each module ends with layer `RELU(inplace=True)`**)

  ```python
  from torchvision import models
  
  vgg_pretrained_features = models.vgg16(pretrained=pretrained).features  # nn.Sequential 31 layers
  # [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
  # number-> ConvModule(conv2d+relu) "M"->Maxpooling
  
  # output with 5 moudle's output (each end with RELU layer)
  vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
  ```

  > What could we learn from this LPIPS implementation？
  >
  > **tear pretrained Module apart to get intermediate layers output** 
  >
  > ```python
  > class vgg16(torch.nn.Module):
  >     def __init__(self, requires_grad=False, pretrained=True):
  >         super(vgg16, self).__init__()
  >         vgg_pretrained_features = models.vgg16(pretrained=pretrained).features  # nn.Sequential hence could be visit by index 
  >         self.slice1 = torch.nn.Sequential()
  >         for x in range(4):
  >             self.slice1.add_module(str(x), vgg_pretrained_features[x]) 
  >             
  >         # ...
  > 	def forward(self, X): 
  >         h = self.slice1(X)
  >         h_relu1_2 = h
  >         # ...
  >         vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
  >         return vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
  >   
  > ```

  

- download file standard ways

  ```python
  def download(url, local_path, chunk_size=1024):
      os.makedirs(os.path.split(local_path)[0], exist_ok=True)
      with requests.get(url, stream=True) as r:
          total_size = int(r.headers.get("content-length", 0))
          with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
              with open(local_path, "wb") as f:
                  for data in r.iter_content(chunk_size=chunk_size):
                      if data:
                          f.write(data)
                          pbar.update(chunk_size)
  ```

- import module from string in config file

  ```python
  import importlib
  
  def get_obj_from_str(string, reload=False):
      module, cls = string.rsplit(".", 1)
      if reload:
          module_imp = importlib.import_module(module)
          importlib.reload(module_imp)
      return getattr(importlib.import_module(module, package=None), cls)
  ```

  

- KL loss

  > [How to generate Gaussian samples](https://medium.com/mti-technology/how-to-generate-gaussian-samples-347c391b7959)
  >
  > - Variance 方差
  > - Standard Deviation
  >   Standard deviation is a statistic that measures the dispersion of a dataset relative to its [mean](https://www.investopedia.com/terms/m/mean.asp) and is calculated as the square root of the [variance](https://www.investopedia.com/terms/v/variance.asp). 
  > - KL divergence [blog](https://zhuanlan.zhihu.com/p/438129018)
  
  `class DiagonalGaussianDistribution` >> `ldm/modules/distributions/distributions.py`
  
  - sample
  
    `logvar` >> $\log{(\sigma^2)}$
    $$
    \text{assume } z\sim\mathcal{N}(0,1) \\
    if \\
    x = \mu + \sigma * Z = \mu + exp(0.5 * \log{(\sigma^2)}) * Z \\
    \text{lets say } x\sim\mathcal{N}(\mu, \sigma)\\
    $$
  
  - KL divergence
    $$
    KL\_Loss = 0.5 *\sum{(\mu^2 + \sigma^2 - 1 - \log{\sigma^2})}
    $$
  
  - VQ-GAN adaptive weight
    $$
    Loss = \mathcal{L}_{resc} +\lambda * \mathcal{L}_{GAN} \\
    \lambda = \frac{\nabla_{GL}{\mathcal{L}_{resc}}}{\nabla_{GL}{\mathcal{L}_{GAN}} +1e-4}
    $$
    **获取最后一层 layer 的梯度**
  
    ```
    last_layer = self.decoder.last_layer.weight
    
    nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
    ```
  
- PatchGAN discriminator

  > [Question: PatchGAN Discriminator](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/39)
  >
  > In fact, a "PatchGAN" is just a convnet! Or you could say all convnets are patchnets: the power of convnets is that they process each image patch identically and independently. 卷积核处理的时候就是一个 image patch

  

## LatentDiffusion Train

>  in `ldm/models/diffusion/ddpm.py`
>  configure file=`configs/latent-diffusion/cin-ldm-vq-f8.yaml`

initialize the `class DDPM(pl.LightningModule)`

### UNetModel

> U-net model in `self.model = DiffusionWrapper(unet_config, conditioning_key)`
> `class UNetModel(nn.Module):`  >> `ldm/modules/diffusionmodules/openaimodel.py`
>
> *The full UNet model with attention and timestep embedding.* 
> See U-net structure >> [Stable-diffusion_U-net-structure-with-Note.pdf](Stable-diffusion_U-net-structure-with-Note.pdf)
>
> - [What is the 'zero_module' used for?](https://github.com/openai/guided-diffusion/issues/21)
>
>    is used to initialize certain modules to zero. 
>
>   ```python
>   def zero_module(module):
>       """
>       Zero out the parameters of a module and return it. >> Initialize the module with 0
>       """
>       for p in module.parameters():
>           p.detach().zero_()
>       return module
>   ```

![U-net structure](./docs/stable_diffusion_unet_architecture.png)

> Upsample 一般都是先 Upsample 再接 `Conv2d(kernel_size=3,stride=1,padding=1)`





#### **timestep_embedding**

> `ldm/modules/diffusionmodules/util.py`
> [timestep embedding blog](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/#what-is-positional-encoding-and-why-do-we-need-it-in-the-first-place) :star:
> [detailed proof for]

**basics**

the time-embedding should satisfy the following criteria:

1. unique each  timestep(word's position)
2. distance between 2 time-step should be consistent across different sentence
3. generalize to longer sentences
4. determinstic

using binary values would be a waste of space in the world of floats. So instead, we can use their float continous counterparts - **Sinusoidal functions. Indeed, they are the equivalent to alternating bits.** 

<figure>
    <img src="https://d33wubrfki0l68.cloudfront.net/ef81ee3018af6ab6f23769031f8961afcdd67c68/3358f/img/transformer_architecture_positional_encoding/positional_encoding.png">
    <figcaption> 
        <center>
        <I>Sinusoidal embedding for dim=50
        </center>
    </figcaption>
</figure>



***sinusoidal timestep embeddings.***

$POS_{i} * POS_{i+k}$ 随着间隔 K 增大，逐渐减小来表示相对位置关系。但内积是对称的，无法区分前后关系

```python
freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
args = timesteps[:, None].float() * freqs[None]  # [N,1] * [1, half] >> [N, half]
embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [N, dim]

time_embed_dim = model_channels * 4
time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )
```



**LitEma**

滑动平均



**register_schedule**

> Whole bunch of formulations could be check at [DDPM video 42:19](https://www.youtube.com/watch?v=y7J6sSO1k50)

- beta >> linear 
  `torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2`

- alpha $[1, 0.9999, \cdots, 7e-4]$

- $\bar{\alpha_i}$ 

  ```
  alphas_cumprod = np.cumprod(alphas, axis=0)
  alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
  ```



## **VQModelInterface**

> `ldm.models.autoencoder.VQModelInterface`



## LitEMA :open_hands:

> [blog: stable-diffusion optimization: EMA weights on CPU](https://lunnova.dev/articles/stable-diffusion-ema-on-cpu/)
> Stable Diffusion 的 LitEMA 滑动平均回到是参数在 GPU 某一时刻加倍，可能导致显存溢出！可以根据此 blog 修改 `LitEMA()` 将参数存在 CPU 上，据说可以从 32G -> 27G 显存

Stable diffusion uses an Exponential Moving Average of the model's weights to improve quality of resulting images and **avoid overfitting to the most recently trained images.**

Stable Diffusion includes an implementation of an EMA called `LitEma`, found at [ldm/modules/ema.py](https://github.com/CompVis/stable-diffusion/blob/69ae4b35e0a0f6ee1af8bb9a5d0016ccb27e36dc/ldm/modules/ema.py)

- How do you implement an EMA for a machine learning model?

  With PyTorch modules you can use [the `named_parameters()` iterator](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.named_parameters) to access all parameters. :star:

- the EMA weights end up on the GPU like everything else. This **doubles the memory required** to store the model parameters! 可能造成显存溢出 !! :warning:

  > [How to keep some LightningModule's parameters on cpu when using CUDA devices for training](https://github.com/Lightning-AI/lightning/issues/3698)



## Text2Image

> `scripts/txt2img.py`

### PLMS

> [PNDM github repo](https://github.com/luping-liu/PNDM)
> [arXiv: Pseudo Numerical Methods for Diffusion Models on Manifolds](https://arxiv.org/abs/2202.09778)
> [local pdf](./2022_ICLR_PNDM_Pseudo-Numerical-Methods-for-Diffusion-Models-on-Manifolds.pdf)

- [invisible-watermark](https://github.com/ShieldMnt/invisible-watermark)



## finetune SDv2-inpaint

> [diffuser baisc training](https://huggingface.co/docs/diffusers/tutorials/basic_training)
>
> SDv2-inpaint checkpoint https://huggingface.co/stabilityai/stable-diffusion-2-inpainting :star:

This `stable-diffusion-2-inpainting` model is resumed from [stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) (`512-base-ema.ckpt`) and trained for another 200k steps.

- Q：输入？

```python
z, c, x, xrec, xc = super().get_input(batch, self.first_stage_key, return_first_stage_outputs=True,force_c_encode=True, return_original_cond=True, bs=bs)
# inpaint: [z: jpg_vae_fea, c:text_fea, x:jpg, xrec: vae_Recon, xc: text]
bchw = z.shape

c_cat = list()  # 
# mask, resize to z's shape
cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
c_cat.append(cc)
# masked_image, apply vae-encde; 
cc = self.get_first_stage_encoding(self.encode_first_stage(cc))
c_cat.append(cc)

c_cat = torch.cat(c_cat, dim=1)
all_conds = {"c_concat": [c_cat], "c_crossattn": [c]}
```

初始噪声 xT，训练时候直接对 GT 加噪

```
prng = np.random.RandomState(seed)
start_code = prng.randn(num_samples, 4, h // 8, w // 8)
start_code = torch.from_numpy(start_code).to(
device=device, dtype=torch.float32)
```

- SDv2-inpaint UNet 输入 [code](https://vscode.dev/github/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py#L1346)

将 noise_fea(c=4), mask(c=1), masked_image_fea(c=4) 合并起来输入 Unet，**调整 unet 的 conv_in 的 C=9 通道**

```
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
```

Framework 参考 BrushNet 的 Controlnet 部分的输入

![ComparsionDiffusionFramework.png](docs/2024_03_Arxiv_BrushNet--A-Plug-and-Play-Image-Inpainting-Model-with-Decomposed-Dual-Branch-Diffusion_Note/ComparsionDiffusionFramework.png)





- Q：SDv2 github repo 代码基于 pytorch lightning，不方便

重写一个 diffuser 的，找一个 diffusers SDv2 的代码 :star:

> issue https://github.com/huggingface/diffusers/issues/1392#issuecomment-1326349638 >> https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

```python
from diffusers import StableDiffusionInpaintPipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "Face of a yellow cat, high resolution, sitting on a park bench"
#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./yellow_cat_on_park_bench.png")
```



- Q：没有 controlnet or Lora 如何 finetune？

> SDv2 inpaint finetune model [code](https://github.com/Stability-AI/stablediffusion/blob/cf1d67a6fd5ea1aa600c4df58e5b47da45f6bdbf/ldm/models/diffusion/ddpm.py#L1504)

只微调一个 input block

```python
                 finetune_keys=("model.diffusion_model.input_blocks.0.0.weight",
                                "model_ema.diffusion_modelinput_blocks00weight"
                                ),
```



