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
  
Instead of using the entire pre-trained VGG model, I used pre-trained weights for the first 10 convolutional layers. The remaining layers were implemented from scratch, with reduced depth (from 4096 to 512), for faster inference. The original VGG network was trained to differentiate between 1000 classes, whereas CARLA provides a maximum of 13. Having 4096 feature maps in the deeper layers seemed unnecessary. 
  
#### Training and Testing
For training I gathered additional data using the CARLA simulator. In total the network was trained on 6400 images gathered from both towns and all 14 weather conditions. Since not all classes are represented equally in the testing images, training loss was calculated using a weighted cross entropy function. Adam optimizer was used for backpropagation. During my research for this challenge I found that stochastic gradient decent performs better over longer epochs. I decided to stick with Adam to due to time limitations during training. 

After each epoch the network was evaluated on hand-picked test data of 500 images. This dataset was selected to contain hard-to-classify images, with small cars and dark/noisy conditions (i.e. hard rain during sunset). The weighted F-score, as described in the challenge, was calculated. Models with the best F-score were saved and retrained with lower learning rates to improve performance.
  
#### Prediction and FPS
Though the challenge involved only classifying vehicles and roads, I decided to segment as many classes as possible. As such my network is able to predict 10 classes. Of the original 13, 6 were combined, resulting in a final total of 10: roads + road lines, buildings + walls, other + traffic signs. The intention behind this was an intuition that the network would be better able to distinguish vehicles and roads if it had a greater classification capacity. This was backed by experimentation which showed improved F-scores (for vehicles and roads) with additional classes in prediction. However, this had a measurable negative effect on inference speed. The intention was to compensate for this loss by employing TensorRT. Unfortunately I experienced significant challenges in implementing TensorRT, and instead had to reduce the input image size to keep FPS above 10. The input image was trimmed 194 pixels from the top and 88 pixel from the bottom. This did not impact the final recall or precision scores, since the trimmed sections should not contain vehicles or roads. Afterwards the image was resized to approximately 2/3 the original resolution. This in particular brought the average f-score down from 0.919 to 0.915, which was my final submitted score.
  
  
## Results


## Enhancements


