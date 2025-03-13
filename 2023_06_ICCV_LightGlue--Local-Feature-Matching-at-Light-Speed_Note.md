# LightGlue: Local Feature Matching at Light Speed

> "LightGlue: Local Feature Matching at Light Speed" ICCV, 2023 Jun
> [paper](http://arxiv.org/abs/2306.13643v1) [code](https://github.com/cvg/LightGlue) [pdf](./2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed.pdf) [note](./2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note.md)
> Authors: Philipp Lindenberger, Paul-Edouard Sarlin, Marc Pollefeys

## Key-point

- Task
- Problems
  - ƥ��Ч�ʵ�

- :label: Label:



## Contributions

- ����������Ѷȣ�ѡ�񲿷�������������Ż���**ʵ�ּ���** ���򵥵� case ֻ��1-2 �� ok���ѵ��ٶ��һЩ layer��

>  One key property is that LightGlue is adaptive to the difficulty of the problem: the inference is much faster on image pairs that are intuitively easy to match, for example because of a larger visual overlap or limited appearance change.

- ƥ����� -> ���� 3D �ؽ��������

> This opens up exciting prospects for deploying deep matchers in latency-sensitive applications like 3D reconstruction.

- ����Ч���� dense matcher LoFTR ��࣬�ٶ����� 8 ��



## Introduction

����ƥ���ͼ������Ӱ��ܴ�

> The most common approach to image matching **relies on sparse interest points that are matched using high-dimensional representations encoding their local visual appearance.** Reliably describing each point is challenging in conditions that exhibit symmetries, weak texture, or appearance changes due to varying viewpoint and lighting. To reject outliers that arise from occlusion and missing points, such representations should also be discriminative. This yields two conflicting objectives, robustness and uniqueness, that are hard to satisfy.

SuperGlue ʹ�� Atten ѧϰƥ�䣬�����Ժܺð�

> It leverages the powerful Transformer model [74] to learn to match challenging image pairs from large datasets. This yields robust image matching in both indoor and outdoor environments. 

ʹ�� Attention ����Ч�ʵ͡���

> however computationally expensive, while the efficiency of image matching is critical for tasks that require a low latency, like tracking, or a high processing volume, like large-scale mapping
>
> Additionally, SuperGlue, as with other Transformer-based models, is notoriously hard to train, requiring computing resources that are inaccessible to many practitioners.

- Q��motivation?

**���� SuperGlue ����Ч�ʣ���ģ�͸Ľ�������ѵ���ɱ�**

> In this paper, we draw on these insights to design LightGlue, a deep network that is more accurate, more efficient, and easier to train than SuperGlue. 
>
> We distill a recipe to train highperformance deep matchers with limited resources, reaching state-of-the-art accuracy within just a few GPU-days.





����Ч���� dense matcher LoFTR ��࣬�ٶ����� 8 ����ʵ������Ч�� & ׼ȷ�Եľ���

> As shown in Figure 1, LightGlue is Pareto-optimal on the efficiency-accuracy trade-off when compared to existing sparse and dense matchers

![fig1](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig1.png)





- Q������Ż��ٶȣ�

ģ�����ദ���˼·��**����������Ѷȣ�ֻ�ò���������������Ż���ʵ�ּ���** ���򵥵� case ֻ��1-2 �� ok���ѵ��ٶ��һЩ layer��

> Unlike previous approaches, LightGlue is adaptive to the difficulty of each image pair, which varies based on the amount of visual overlap, appearance changes, or discriminative information. 
>
> , a behavior that **is reminiscent of** how humans process visual information. 

����һЩ layer ��ģ���Լ�ȥ����Ƿ�Ҫ�������� or ����һЩ��

> This is achieved by 1) predicting a set of correspondences after each computational blocks, and 2) enabling the model to **introspect them and predict whether further computation is required.** LigthGlue also discards at an early stage points that are not matchable, thus focusing its attention on the covisible area.

![fig2](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig2.png)



֧�� plug&play �� SLAM���������� feature sets ����Ԥ���ǿ�Ĺ�����

> Our experiments show that LightGlue is a plug-and-play replacement to SuperGlue: **it predicts strong matches from two sets of local features**, at a fraction of the run time.





### Deep matchers

- "SuperGlue: Learning Feature Matching with Graph Neural Networks" CVPR, 2019 Nov 26
  [paper](http://arxiv.org/abs/1911.11763v2) [code](https://github.com/magicleap/SuperGluePretrainedNetwork.) [pdf](./2019_11_CVPR_SuperGlue--Learning-Feature-Matching-with-Graph-Neural-Networks.pdf) [note](./2019_11_CVPR_SuperGlue--Learning-Feature-Matching-with-Graph-Neural-Networks_Note.md)
  Authors: Paul-Edouard Sarlin, Daniel DeTone, Tomasz Malisiewicz, Andrew Rabinovich

![fig1](docs/2019_11_CVPR_SuperGlue--Learning-Feature-Matching-with-Graph-Neural-Networks_Note/fig1.png)

> The first of its kind, SuperGlue [56] combines the expressive representations of Transformers [74] with optimal transport [48] to solve a partial assignment problem.
>
> It learns powerful priors about scene geometry and camera motion and is thus robust to extreme changes and generalizes well across data domains. 

SuperGlue ���ŵ������ӣ����Ӷ�̫��

> Inheriting the limitations of early Transformers, SuperGlue is hard to train and its complexity grows quadratically with the number of keypoints.



LoFTR ����һ�����ܵ� grid ��Ϊ�ο��㣬���ܺúܶ࣬���������ˡ�����

> Conversely, dense matchers like LoFTR [68] and followups [9, 78] match points distributed on dense grids rather than sparse locations. This boosts the robustness to impressive levels but is generally much slower because it processes many more elements. 
>
> This limits the resolution of the input images and, in turn, the spatial accuracy of the correspondences. While LightGlue operates on sparse inputs, we show that fair tuning and evaluation makes it competitive with dense matchers, for a fraction of the run time
>
> - "LoFTR: Detector-Free Local Feature Matching with Transformers" CVPR, 2021 Apr 1
>   [paper](http://arxiv.org/abs/2104.00680v1) [code]() [pdf](./2021_04_CVPR_LoFTR--Detector-Free-Local-Feature-Matching-with-Transformers.pdf) [note](./2021_04_CVPR_LoFTR--Detector-Free-Local-Feature-Matching-with-Transformers_Note.md)
>   Authors: Jiaming Sun, Zehong Shen, Yuang Wang, Hujun Bao, Xiaowei Zhou





- Q��ͼ��ֲ�����զ���ģ�

**local features���ؼ���λ�ð������� �� detection confidence��visual descriptor ʹ�� SIFT �� SuperPoint ���Ƶ�  CNN ��ȡ����**

> Positions consist of x and y image coordinates as well as a detection confidence c, pi := (x, y, c)i .
>
> Visual descriptors di �� R D can be those extracted by a CNN like SuperPoint or traditional descriptors like SIFT

�� SuperGlue �������棬ʹ�� ��λ�ã�visual descriptor��

> Images A and B have M and N local features, indexed by A := {1, ..., M} and B := {1, ..., N}, respectively.
>
> We design LightGlue to output a set of correspondences M = {(i, j)} ? A �� B
>
> Each point is matchable at least once, as it stems from a unique 3D point, and some keypoints are unmatchable, due to occlusion or non-repeatability





- Q����ζ� SuperGlue ���٣�

���ٵ���������������

> Subsequent works make it more efficient by reducing the size of the attention mechanism

���������Ѷȣ���̬���� "�����С"

> LightGlue instead brings large improvements for typical operating conditions, like in SLAM, without compromising on performance for any level of difficulty. This is achieved by dynamically **adapting the network size instead of reducing its overall capacity.**



## methods

![fig3](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig3.png)

���ѵ������磬ÿ����ʾ��ȥ�����������Ż���

> LightGlue is made of a stack of L identical layers that process the two sets jointly. Each layer is composed of self- and cross-attention units that update the representation of each point.

�����һ����ǰ�˳����ƣ������ǰ layer �������㹻�þ��ˣ������û��Ҫ���˰ɡ�

> A confidence classifier c helps decide whether to stop the inference. If few points are confident, the inference proceeds to the next layer but we prune points that are confidently unmatchable.



### Adaptive depth and width

> - "Confident Adaptive Language Modeling" NIPS
> - "Depth-Adaptive Transformer" ICLR
> - "BranchyNet: Fast inference via early exiting from deep neural networks" ICPR
> - "Anytime Stereo Image Depth Estimation on Mobile Devices" ICRA
> - "Anytime Dense Prediction with Confidence Adaptivity" ICLR



- Q��������������㹻�ã�:star:

����һ�� MLP Ԥ��ÿ���������Ƿ�ɿ���

> We add two mechanisms that avoid unnecessary computations and save inference time
>
> . At the end of each layer, LightGlue infers the confidence of the predicted assignment of each poi

![eq9](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/eq9.png)

![eq10](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/eq10.png)

> We observe, as in [62], that the classifier itself is less confident in early layers

exit δ�ﵽ��һЩ�㣬��Ϊû�ã��Ͷ��������͵���

> Point pruning: When **the exit criterion is not met**, points that are predicted as both confident and unmatchable are unlikely to aid the matching of other points in subsequent layers. Such points are for example in areas that are clearly not covisible across the images. We therefore discard them at each layer and feed only the remaining points to the next one. This significantly reduces computation, given the quadratic complexity of attention, and does not impact the accuracy



���ֿ��Լ���ѵ��

![fig5](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig5.png)



### Comparison with SuperGlue

- Q���� SuperGlue ����

Attention �������λ�ñ���

LightGlue �����ƶȺͿ�ƥ��ȷֽ⿪��ר�Ÿ�һ�� MLP ȥԤ��ƥ���

> SuperGlue predicts an assignment by solving a differentiable optimal transport problem using the Sinkhorn algorithm [66, 48]. It consists in many iterations of row-wise and column-wise normalization, which is expensive in terms of both compute and memory. SuperGlue adds a dustbin to reject unmatchable points
>
> **We found that the dustbin entangles the similarity score of all points and thus yields suboptimal training dynamics**. LightGlue disentangles similarity and matchability, which are much more efficient to predict

LightGlue �����������ÿ�� layer �� confidence �����㣬���ж� exit

> The lighter head of LightGlue makes it possible to predict an assignment at each layer and to supervise it. This speeds up the convergence and enables exiting the inference after any layer, which is key to the efficiency gains of LightGlue



## setting

> Our experiments show that LightGlue is a plug-and-play replacement to SuperGlue

- Q��ѵ�� trick��

exit ����ɸ��һ���� unmatched �㣬����ѵ��

> : While the LightGlue architecture improves the training speed, stability, and accuracy, we found that some details have a large impact too. 

���� interest points ����

> Training with more points also does: we use 2k per image instead of 1k.

һ�� 24G VRAM mixed-precision + gradientCheckpoint ʵ�� bs=32 :star:

> The batch size matters: we use gradient checkpointing [10] and mixed-precision to fit 32 image pairs on a single GPU with 24GB VRAM



- Q������ṹ��

> Implementation details: LighGlue has L=9 layers. Each attention unit has 4 heads. All representations have dimension d=256. Throughout the paper, run-time numbers labeled as optimized use an efficient implementation of selfattention [14]. More details are given in the Appendix.



## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

��ǰ exit �ĵ��ע��ɫ��������ȷʵ�е��ã���ʵ�ּ��١�

![fig4](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig4.png)

?	



Ч���� LoFTR �ӽ���

![tb1](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/tb1.png)

���ٿ��Լ���

![tb2](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/tb2.png)



չʾ��ͬ�Ѷȵ�ƥ��ͼ��adaptive-width �Ƕ�̬���� interest point ���������ڲ���ƥ��ĵ㣩��adaptive depth ���� layers ��

![fig8](docs/2023_06_ICCV_LightGlue--Local-Feature-Matching-at-Light-Speed_Note/fig8.png)



## Limitations

## Summary :star2:

> learn what & how to apply to our task

- Q���ܷ�������ֻ������ʵ�����٣����ټ��벽��

- Q���ֱ�����ɶӰ���4K һ���죿

