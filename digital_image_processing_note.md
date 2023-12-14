# Image Processing

> [opencv official tutorial](https://docs.opencv.org/3.4/d9/df8/tutorial_root.html)



## edge filter

> [opencv official tutorial: Laplace](https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html) :+1:



**Sobel Operator**

> [opencv tutorial](https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html)

x,y 方向的两个 kernel $G_x = K_x * I, ~G_y = K_y * I$



**Laplacian Filter**

> [Laplacian filter blog](https://medium.com/@rajilini/laplacian-of-gaussian-filter-log-for-image-processing-c2d1659d5d2)

The Laplacian of an image highlights regions of rapid intensity change. Any feature with a sharp discontinuity will be enhanced by a Laplacian operator.
It is a **second-order filter** used in image processing for edge detection and feature extraction.

![](https://miro.medium.com/v2/resize:fit:720/format:webp/1*a4dUIi_OmkE6_T5_JjkvCg.png)



![](https://www.nv5geospatialsoftware.com/docs/html/images/laplacian.gif)


```python
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(r"E:\eye.png", cv2.IMREAD_COLOR)
image = cv2.GaussianBlur(image, (3, 3), 0)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered_image = cv2.Laplacian(image_gray, cv2.CV_16S, ksize=3)
```



**Canny**

> [opencv tutorial](https://docs.opencv.org/3.4/da/d5c/tutorial_canny_detector.html)

1. 计算 strength 类似 sobel 算子，和角度
2. NMS 非极大抑制，判断是否为边缘