# Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models

> "Zero-Shot Video Editing Using Off-The-Shelf Image Diffusion Models" Arxiv, 2023 Mar, vid2vid-zero :star:
> [paper](https://arxiv.org/abs/2303.17599) [code](https://github.com/baaivision/vid2vid-zero?utm_source=catalyzex.com)
> [paper local pdf](./2023_03_Arxiv_ Zero-shot-video-editing-using-off-the-shelf-image diffusion-models.pdf)

## Key-point

- Video editing with off-the-shelf image diffusion models.

- No training on any video.

  参考下重建 video

- Temporal consistency >> cross-attention maps



**Contributions**

## Related Work

## methods

![](https://github.com/baaivision/vid2vid-zero/raw/main/docs/vid2vid-zero.png)



## Experiment

## **Limitations**

显存消耗很大！8 帧的视频（1s fps=8）24G显存，10s fps=30的视频需要



## Code Implementation

- `class AutoencoderKL(ModelMixin, ConfigMixin)`
  *Variational Autoencoder (VAE) model with KL loss from the paper Auto-Encoding Variational Bayes by Diederik P.*

  `latents = vae.encode(pixel_values).latent_dist.sample()`

- `class NullInversion:`

  `ddim_inv_latent, uncond_embeddings = null_inversion.invert(...)`

- `class AttentionRefine(AttentionControlEdit)`

  - 统计 cross-attention 模块个数

    ```python
        def register_recr(net_, count, place_in_unet):
            if net_.__class__.__name__ == 'CrossAttention':
                net_.forward = ca_forward(net_, place_in_unet)
                return count + 1
            elif hasattr(net_, 'children'):
                for net__ in net_.children():
                    count = register_recr(net__, count, place_in_unet)
            return count
        
        cross_att_count = 0
        # sub_nets = model.unet.named_children()
        # we take unet as the input model
        sub_nets = model.named_children()  # [name_str, module]
    ```




## Summary :star2:

> learn what & how to apply to our task