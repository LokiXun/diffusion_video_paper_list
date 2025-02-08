# Simulating Fluids in Real-World Still Images

> "Simulating Fluids in Real-World Still Images" ICCV-2023-oral, 2022 Apr 24
> [paper](http://arxiv.org/abs/2204.11335v1) [code](https://github.com/simon3dv/SLR-SFS) [web](https://slr-sfs.github.io/) [pdf](./2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images.pdf) [note](./2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note.md)
> Authors: Siming Fan, Jingtan Piao, Chen Qian, Kwan-Yee Lin, Hongsheng Li

## Key-point

- Task
- Problems
- :label: Label:

## Contributions

## Introduction

- Q：先前工作背景不该动的物体，也跟着水流移动？

预测水流的 RGBA 图层，最后合在一起





## methods

![fig1](docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/fig1.png)

给一个 spare motion 做为 hint 预测水流 motion；

预测 RGBA，把水流的图层分割出来

<img src="docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/fig2.png" alt="fig2" style="zoom: 80%;" />



- Q：水流不真实？

使用 NS 公式计算真实的水流移动

![image-20250207201749428](docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/image-20250207201749428.png)

生成背景 3D mesh, 再用 NS 公式渲染

### motion calc

![fig4](docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/fig4.png)



构造 3D mesh，加上 NS 仿真水流移动轨迹，去优化水流轨迹

![fig17](docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/fig17.png)

### Data

收集水流视频构造了一个训练数据集



## setting

## Experiment

> ablation study 看那个模块有效，总结一下

![tb1](docs/2022_04_ICCV_Simulating-Fluids-in-Real-World-Still-Images_Note/tb1.png)

效果没差多少啊



## Limitations

https://slr-sfs.github.io/  视频里面背景也在抖动





## Summary :star2:

> learn what



### how to apply to our task

- 预测水流的 RGBA 图层，分割背景 & 水流，看效果还可以啊
- 构造 3D mesh，加上 NS 仿真水流移动轨迹，去优化水流轨迹
