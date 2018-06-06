# Lyft's Perception Challenge

### Table of Contents
  1) Introduction
  2) Related Work
  3) Implementation
  4) Results
  5) Enhancements

[image1]: ./images/network.png "Network architecture"
[image2]: ./images/good.png "Good prediction results"
[image3]: ./images/bad.png "Bad prediction results"

## Introduction
This challenge involved producing pixel-by-pixel annotations of vehicles and roads in images from the [CARLA simulator](www.carla.org), referred to as semantic segmentation. A custom built Fully Convolutional Network (FCN) was employed for this task. Dilated convolutions were used in place of deeper max-pooling to increase the network’s receptive field, while maintaining granular image details. A skip connection was added to the network in order to preserve finer details lost during earlier max-pooling. The resulting architecture ranked 13th in the competition with a final score of 91.49. However, this score is not an accurate representation of the networks performance. Due to challenges in implementing TensorRT, input image dimensions were reduced to improve inference speed at the cost of recall and precision.

## Related Work
Since this was my first time implementing a network for semantic segmentation, I closely followed the designs outlined in these two papers:
  
1)	[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
The authors took a pre-existing VGG network and replaced all fully connected layers with convolutional layers. The resulting network was trained on pixel-wise annotations to produce segmentation maps. As a starting point I adopted their approach, using a pre-trained VGG with convolutional replacement and up-sampled skip connections. The results were impressive but coarse due to loss of finer details during max-pooling.
  
2)	[Multi-Scale Context Aggregation by Dilated Convolutions]( https://arxiv.org/abs/1511.07122)
In order to produce more granular segmentations I redesigned my network to use dilated convolutions as described in the paper above. The last max-pooling layer was replaced with multiple dilated convolutions, increasing the networks receptive field while preserving granular details. Unlike the author’s design, I kept one skip connection (from the third convolutional stack) to improve precision and recall. This skip connection was scaled by a factor of 0.01 to give preferences to outputs from the deeper layers, and avoid larger weight updates to the shallow layers.


## Implementation
#### Network Architecture
My final network contained 19 layers (13 regular convolutions, 4 dilated convolutions, and 2 transposed convolutions), illustrated in the image below. Note the output node contains the last transposed convolution.
  
![alt text][image1]

## Results


## Enhancements


