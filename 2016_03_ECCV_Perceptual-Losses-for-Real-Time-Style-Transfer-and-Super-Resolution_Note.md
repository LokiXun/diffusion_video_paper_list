# Perceptual Losses for Real-Time Style Transfer and Super-Resolution

> "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" ECCV, 2016 Mar
> [paper](https://arxiv.org/abs/1603.08155) [code](https://github.com/DmitryUlyanov/texture_nets) 
> [blog](https://iq.opengenus.org/vgg19-architecture/)
> Authors: Justin Johnson, Alexandre Alahi, Li Fei-Fei

## Key-point

- Task: ���Ǩ��
- Background
- :label: Label:

## Contributions

## Related Work

### **VGG network**

> "Very Deep Convolutional Networks for Large-Scale Image Recognition" ICLR, 2014 Sep
> [paper](https://arxiv.org/pdf/1409.1556.pdf)

Conv Layers: `3x3 Conv` + `2x2 MaxPooling`(����ÿ�� conv ���涼��) + `Relu`(All hidden layers are equipped with the rectification non-linearity ���� conv layer ���涼�ӣ���֤������)

three Fully-Connected (FC) : channel 4096, 4096, 1000(������) 

���� VGG19(����� E) 16���� + 3FC���� 5 �� stage

![image-20231026124102444](docs/2016_03_ECCV_Perceptual-Losses-for-Real-Time-Style-Transfer-and-Super-Resolution_Note/VGG19architecture.png)

VGG19 �� 5 �� stage��ÿ��stage ������������ֱ�Ϊ 2,2,4,4,4 һ�� 16 ����



## methods

ʹ���� ImageNet ��ѵ���õ��� VGG16 ��Ϊ Loss Network

### Perceptual loss

> Rather than encouraging the pixels of the output image $\hat{y} = f_W (x)$ to exactly match the pixels of the target image y, we instead **encourage them to have similar feature representations** as computed by the loss network ��
>
> Using a feature reconstruction loss for training our image transformation networks **encourages the output image $\hat{y}$ to be perceptually similar to the target image y, but does not force them to match exactly.**

**feature reconstruction loss is the (squared, normalized) Euclidean distance** 
, $\phi_j$ Ϊ VGG16 �� j ��ļ������Relu�����������ͼ��СΪ $\begin{aligned}C_j\times H_j\times W_j\end{aligned}$
$$
\ell_{feat}^{\phi,j}(\hat{y},y)=\frac1{C_jH_jW_j}\|\phi_j(\hat{y})-\phi_j(y)\|_2^2
$$
ʹ�� VGG ���� loss ��Ӽ�Ȩ����Ϊ���� loss
$$
W^*=\arg\min_W\mathbf{E}_{x,\{y_i\}}\left[\sum_{i=1}\lambda_i\ell_i(f_W(x),y_i)\right]
$$


- С��

  ![image-20231026140447773](docs/2016_03_ECCV_Perceptual-Losses-for-Real-Time-Style-Transfer-and-Super-Resolution_Note/VGG_diff_layer_output.png)

  1. ��ȡVGG16 ǳ��������������ͼ���ϸ�ڣ��ؽ�������ͼ���Ŀ��ͼ���������֣�
     �����޸� or �ؽ�����ʹ�� VGG ǳ������
  2. ��ȡ������������ڸ߼�������Ϣ������ͼ������ݡ�����ռ�ṹ������ɫ������������״��ϸ�ڶ�ʧ



### Style Reconstruction Loss

Gram matrix ����
$$
G_j^\phi(x)_{c,c^{\prime}}=\frac1{C_jH_jW_j}\sum_{h=1}^{H_j}\sum_{w=1}^{W_j}\phi_j(x)_{h,w,c}\phi_j(x)_{h,w,c^{\prime}}.
$$
ʹ���� Gram matrix ���ò�ͬ�ߴ����룬��Ϊ Gram matrices �ߴ�һ��

�ö��������� loss ������

> perform style reconstruction from a set of layers J



## Usage :star:

![image-20231026124102444](docs/2016_03_ECCV_Perceptual-Losses-for-Real-Time-Style-Transfer-and-Super-Resolution_Note/VGG19architecture.png)

> - [Bring old films code](https://github.com/raywzy/Bringing-Old-Films-Back-to-Life/blob/54df59417c9c428c59d38327cc8eb424af6b4b0c/VP_code/models/loss.py#L77)  ��Ƶ�޸�����
>
> - [UEGAN code](https://github.com/eezkni/UEGAN/blob/2c729ee8b8daaf600aee6dc0779c14286e7fd768/losses.py#L12) ͼ����ǿ���񣬵���������Ϣ
>
>   ʹ�� `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1` , `relu5_1` Ȩֵ [1.0/64, 1.0/64, 1.0/32, 1.0/32, 1.0/1]
>
>   ģ�������Χ [-1,1]�� ����Ҫ����һ�� >> **perceptual loss ʹ�õ� VGG16 �� ImageNet ��ѵ�����ȹ�һ�� [0, 1]���ڰ�ImageNet ��ֵ�����׼��** :warning:

Perceptual Loss ������ feature reconstruction loss at layer **relu2_2** and style reconstruction loss at layers **relu1_2, relu2_2, relu3_3, and relu4_3** of the VGG-16 loss network



### torchvision VGG ʵ��

> `torchvision\models\vgg.py` 

```python
cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}
```

E ���� VGG19������Ϊͨ������MΪ max-pooling����Ӧ VGG19 �Ľṹͼ����VGG19 ǰ�������֣�

```python
def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)
```

ʹ��ʱ���� `batch_norm`��ÿ�� Conv layer Ϊ `[conv2d, nn.ReLU(inplace=True)]`

> `nn.Relu(inplace=True)` inplace ���ã�
> [torch doc](https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html#relu)  optionally do the operation in-place. 
> [See Blog](https://blog.csdn.net/manmanking/article/details/104830822) �� **`inplace = False` ʱ,�����޸���������ֵ,���Ƿ���һ���´����Ķ���,**���Դ�ӡ��**����洢��ַ**��ͬ��`inplace = True` ʱ,���޸���������ֵ,���Դ�ӡ������洢��ַ��ͬ,

VGG19 ���3�� FC

```python
Class VGG(nn.Module):
    def __init__(...):
        # ...
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        # ...
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```



### example

> old-films �����ʵ��

```python
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
                
    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
```

��Ӧ `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, `relu5_1` �������Ȩֵ�� `[1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]` �ĳɾ�Ϊ 1 PSNR && ������ֵ������� 0.6dB



## Experiment

> ablation study ���Ǹ�ģ����Ч���ܽ�һ��

### SISR

minimizing feature reconstruction loss at layer `relu2_2` from the VGG-16 loss network ��

- report PSNR and SSIM [54], computing both only on the Y channel after converting to the YCbCr colorspace

  PSNR, SSIM ���������˶�ͼ�������Ĺ۸�

  > The traditional metrics used to evaluate super-resolution are PSNR and SSIM [54], both of which have been found to correlate poorly with human assessment of visual quality

  

## Limitations

## Summary :star2:

> learn what & how to apply to our task

