# VividPast - Automated Video Colorizer
## Ryan Green, Aayush Jain, Justine Lo
### CS 6140 - Machine Learning - Fall 2018 - Northeastern University

Recently, the colorization of photos through supervised machine learning models has suggested a potential solution to the painstaking process of manually coloring black and white photos and videos. Though promising, the automation of this task remains a difficult problem that suggests the need for multi- leveled image analysis. This project proposed a solution by combining a custom-trained con- volutional neural network (CNN) with semantic image segmentation information, combining low level features with high level semantic information. The extraction of high level features provide additional information about image contents to help with colorization. The fusion model showed a much greater re- duction of loss over the training period, as compared to the CNN model without the image segmentation information. The colored predictions of the fusion model also showed realistic results during validation. The fusion of high level semantic information in the CNN seemed to have enhanced the overall color prediction of grayscale images.
