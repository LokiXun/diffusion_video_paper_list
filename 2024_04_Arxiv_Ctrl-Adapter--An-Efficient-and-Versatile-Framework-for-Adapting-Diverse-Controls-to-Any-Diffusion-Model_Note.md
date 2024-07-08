# Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model

> "Ctrl-Adapter: An Efficient and Versatile Framework for Adapting Diverse Controls to Any Diffusion Model" Arxiv, 2024 Apr 15
> [paper](http://arxiv.org/abs/2404.09967v2) [code](https://github.com/HL-hanlin/Ctrl-Adapter) [website](https://ctrl-adapter.github.io/) [pdf](./2024_04_Arxiv_Ctrl-Adapter--An-Efficient-and-Versatile-Framework-for-Adapting-Diverse-Controls-to-Any-Diffusion-Model.pdf) [note](./2024_04_Arxiv_Ctrl-Adapter--An-Efficient-and-Versatile-Framework-for-Adapting-Diverse-Controls-to-Any-Diffusion-Model_Note.md)
> Authors: Han Lin, Jaemin Cho, Abhay Zala, Mohit Bansal

## Key-point

- Task
- Problems
- :label: Label:

支持多种 condition 输入！

## Contributions

## Introduction

## methods

- Q：image condition 特征如何提取？

方案一、参考 PASD 训练 RGB loss



- Q：image2video 的 condition 如何处理？是先 repeat 输入 controlnet 还是？



- Q：code ？

> https://github.com/HL-hanlin/Ctrl-Adapter?tab=readme-ov-file#controllable-video-generation

```
python inference.py \
--model_name "i2vgenxl" \
--control_types "depth" \
--huggingface_checkpoint_folder "i2vgenxl_depth" \
--eval_input_type "frames" \
--evaluation_input_folder "assets/evaluation/frames" \
--n_sample_frames 16 \
--extract_control_conditions True \
--num_inference_steps 50 \
--control_guidance_end 0.8 \
--height 512 \
--width 512
```



**i2vgenxl pipeline**

> https://github.com/HL-hanlin/Ctrl-Adapter/blob/main/i2vgen_xl/pipelines/i2vgen_xl_controlnet_adapter_pipeline.py#L119

关注 image 如何处理即可

```python
                i2vgenxl_outputs = pipe(
                    prompt=prompt,
                    negative_prompt="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
                    height = inference_args.height, 
                    width = inference_args.width,
                    image= images_pil[0],  # initial frame
                    control_images = control_images,  # depths, edge image
                    num_inference_steps=inference_args.num_inference_steps,
                    guidance_scale=9.0, 
                    generator=generator,
                    target_fps = target_fps,
                    num_frames = num_frames,
                    output_type="pil",
                    **kwargs
                ) 
```

> huggingface 模型权重：https://huggingface.co/ali-vilab/i2vgen-xl



**Conditions 提取特征**

> https://github.com/HL-hanlin/Ctrl-Adapter/blob/main/i2vgen_xl/pipelines/i2vgen_xl_controlnet_adapter_pipeline.py#L751

`self.control_image_processor.preprocess` 预处理包含 resize，normalize; 

把多帧 conditions 在 c 维度 `concat`

```python
    def prepare_images(
        self,
        images,
        width,
        height,
        batch_size,
        num_images_per_prompt,
        device,
        dtype,
        do_classifier_free_guidance=False,
        guess_mode=False,
    ):
        images_pre_processed = [self.control_image_processor.preprocess(image, height=height, width=width).to(dtype=torch.float32) for image in images]

        images_pre_processed = torch.cat(images_pre_processed, dim=0)

        repeat_factor = [1] * len(images_pre_processed.shape)
        repeat_factor[0] = batch_size * num_images_per_prompt
        images_pre_processed = images_pre_processed.repeat(*repeat_factor)

        images = images_pre_processed.unsqueeze(0)
        images = images.to(device=device, dtype=dtype)

        if do_classifier_free_guidance and not guess_mode:
            repeat_factor = [1] * len(images.shape)
            repeat_factor[0] = 2
            images = images.repeat(*repeat_factor)

        return images

```



 **Image First Frame**

提取 CLIP 特征 `image_embeddings` ：`image_encoder: CLIPVisionModelWithProjection,` CLIP-image encoder 加上一个 MLP，从 `"hidden_size": 1280,` ->  `"projection_dim": 1024,`

> https://vscode.dev/github/HL-hanlin/Ctrl-Adapter/blob/maindeling_clip.py#L1255
>
> 预训练 CLIP config: https://huggingface.co/ali-vilab/i2vgen-xl/blob/main/image_encoder/config.json

```python
class CLIPVisionModelWithProjection(CLIPPreTrainedModel):
    config_class = CLIPVisionConfig
    main_input_name = "pixel_values"

    def __init__(self, config: CLIPVisionConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionTransformer(config)

        self.visual_projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(CLIP_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPVisionModelOutput, config_class=CLIPVisionConfig)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPVisionModelOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output

        image_embeds = self.visual_projection(pooled_output)

        if not return_dict:
            outputs = (image_embeds, vision_outputs[0]) + vision_outputs[2:]
            return tuple(output for output in outputs if output is not None)

        return CLIPVisionModelOutput(
            image_embeds=image_embeds,
            last_hidden_state=vision_outputs.last_hidden_state,
            hidden_states=vision_outputs.hidden_states,
            attentions=vision_outputs.attentions,
        )
```



`prepare_image_latents` 提取各帧 VAE 特征 `image_latents`；剩下的帧用全 1 的 mask

> https://vscode.dev/github/HL-hanlin/Ctrl-Adapter/blob/main/i2vgen_xl/pipelines/i2vgen_xl_controlnet_adapter_pipeline.py#L487

```python
    def prepare_image_latents(
        self,
        image,
        device,
        num_frames,
        num_videos_per_prompt,
    ):
        image = image.to(device=device)
        image_latents = self.vae.encode(image).latent_dist.sample()
        image_latents = image_latents * self.vae.config.scaling_factor

        # Add frames dimension to image latents
        image_latents = image_latents.unsqueeze(2)

        # Append a position mask for each subsequent frame
        # after the intial image latent frame
        frame_position_mask = []
        for frame_idx in range(num_frames - 1):
            scale = (frame_idx + 1) / (num_frames - 1)
            frame_position_mask.append(torch.ones_like(image_latents[:, :, :1]) * scale)
        if frame_position_mask:
            frame_position_mask = torch.cat(frame_position_mask, dim=2)
            image_latents = torch.cat([image_latents, frame_position_mask], dim=2)

        # duplicate image_latents for each generation per prompt, using mps friendly method
        image_latents = image_latents.repeat(num_videos_per_prompt, 1, 1, 1, 1)

        if self.do_classifier_free_guidance:
            image_latents = torch.cat([image_latents] * 2)

        return image_latents
```







## Experiment

> ablation study 看那个模块有效，总结一下

## Limitations

## Summary :star2:

> learn what & how to apply to our task

