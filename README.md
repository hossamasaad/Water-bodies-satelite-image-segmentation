# Water Bodies Satelite Image Segmentation
![GitHub repo size](https://img.shields.io/github/repo-size/hossamasaad/Water-bodies-satelite-image-segmentation)
![GitHub contributors](https://img.shields.io/github/contributors/hossamasaad/Water-bodies-satelite-image-segmentation)
![GitHub stars](https://img.shields.io/github/stars/hossamasaad/Water-bodies-satelite-image-segmentation?style=social)
![GitHub forks](https://img.shields.io/github/forks/hossamasaad/Water-bodies-satelite-image-segmentation?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/hossamasaad10?style=social)

Apply Semantic segmenation techniques such as UNet, DeepLab and FCN to water bodies satelite images and deploy using flask and streamlit

## Dataset - [Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies)
A collection of water bodies images captured by the Sentinel-2 Satellite. Each image comes with a black and white mask where white represents water and black represents something else but water. The masks were generated by calculating the NWDI (Normalized Water Difference Index) which is frequently used to detect and measure vegetation in satellite images, but a greater threshold was used to detect water bodies.


## How to try it?

## Architectures

### 1. U-Net
![U-Net](assets/unet.png)

### 2. FCN-8 (Fully Convolutional Network)
![FCN](assets/fcn.jpg)

### 3. SegNet
![SegNet](assets/segnet.png)

### 4. ShelfNet
![ShelfNet](assets/shelfnet.png)

### 5. DeepLabV3
![DeepLab](assets/deeplab.png)

## Results
|Model|Loss (binary cross entropy)|Dice|iou|epochs|
|--|--|--|--|--|
|U-Net|0.3040|0.7205|0.5649|50|
|FCN-8|0.3683|0.5725|0.4013|50|
|DeepLabv3|-|-|-|-|

![UNet Results](assets/unet_result.png)
![UNet Results](assets/unet_result2.png)

## Project Structure


## Tools
- Python
- Tensorflow
- streamlit
- flask

## References
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597v1)
- [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/abs/1605.06211v1)
- [SegNet: A Deep Convolutional Encoder-Decoder Architecture for Image Segmentation](https://arxiv.org/abs/1511.00561v3.pdf)
- [ShelfNet for Fast Semantic Segmentation](https://arxiv.org/abs/1811.11254v6)
- [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/abs/1706.05587v3)
