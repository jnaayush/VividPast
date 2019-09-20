# VividPast - Automated Video Colorizer
## Ryan Green, Aayush Jain, Justine Lo
### CS 6140 - Machine Learning - Fall 2018 - Northeastern University

Recently, the colorization of photos through supervised machine learning models has suggested a potential solution to the painstaking process of manually coloring black and white photos and videos. Though promising, the automation of this task remains a difficult problem that suggests the need for multi- leveled image analysis. This project proposed a solution by combining a custom-trained con- volutional neural network (CNN) with semantic image segmentation information, combining low level features with high level semantic information. The extraction of high level features provide additional information about image contents to help with colorization. The fusion model showed a much greater re- duction of loss over the training period, as compared to the CNN model without the image segmentation information. The colored predictions of the fusion model also showed realistic results during validation. The fusion of high level semantic information in the CNN seemed to have enhanced the overall color prediction of grayscale images.

## Introduction

Historically, the colorization of black and white images has been a slow, painstaking process requiring skilled human colorists. This process has made the colorization of black and white motion pictures an unthinkably meticulous and labor-intensive problem. Recent advances in computer vision and neural nets has opened the door to new, advanced forms of image and video analysis, and we propose to apply these techniques to the previously onerous problem of colorizing monochrome pictures.

## Technical Approach

The problem of automated image colorization is not a simple one. While the task can be formulated as a curious type of data imputation in which color values are predicted or ”hallucinated” given grayscale values, the mapping from a grayscale channel to color channels is anything but deterministic. There is a wide range of color values that a single grayscale value might take on, so contextual and semantic information is essential to making realistic guesses.
 #### Loss Function

Our loss function for each image’s prediction is defined as follows: 
$$
J_{img}(L,Y)= \sum_{x∈W} \sum_{y∈H} \sum_{c∈\{A,B\}} |h_c(L_{x,y})−Y_{c_{x,y}}|
$$

#### Network Architecture
![NW](/media/NW.jpg "Netowrk Architecture")

#### Convolution Auto-Encoder

![NW](/media/NW1.jpg "Netowrk Architecture")

#### Results

![R](/media/R1.jpg "Result")

![R](/media/R2.jpg "Result")

