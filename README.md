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
This challenge involved producing pixel-by-pixel annotations of vehicles and roads in images from the [CARLA simulator](www.carla.org), referred to as semantic segmentation. A custom built Fully Convolutional Network (FCN) was employed for this task. Dilated convolutions were used in place of deeper max-pooling to increase the network’s receptive field, while maintaining granular image details. A skip connection was added to the network in order to preserve finer details lost during earlier max-pooling. The resulting architecture ranked 13th in the competition with a final score of 91.49.

## Related Work
This was my first time implementing semantic segmentation and I closely followed the designs outlined in the following papers:
  
1)	[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)  
As a starting point I adopted their approach, using a pre-trained VGG with convolutional replacement and up-sampled skip connections. The results were impressive but coarse due to loss of finer details during max-pooling.
  
2)	[Multi-Scale Context Aggregation by Dilated Convolutions]( https://arxiv.org/abs/1511.07122)  
In order to produce more granular segmentations I redesigned my network with dilated convolutions. The last max-pooling layer was replaced with multiple dilated convolutions, increasing the networks receptive field while preserving granular details. Unlike the authors' design, I kept one skip connection (from the third convolutional stack) to improve precision and recall. This skip connection was scaled by a factor of 0.01 to give preferences to outputs from the deeper layers, and avoid larger weight updates to the shallow layers.


## Implementation
#### Network Architecture
The final network design contained 19 layers (13 regular convolutions, 4 dilated convolutions, and 2 transposed convolutions), illustrated in the image below. Note the output node contains the last transposed convolution.
   
![alt text][image1]
  
Instead of using the entire pre-trained VGG model, I used pre-trained weights for the first 10 convolutional layers. The remaining layers were implemented from scratch, with reduced depth (from 4096 to 512), for faster inference. The original VGG network was trained to differentiate between 1000 classes, whereas CARLA provides a maximum of 13. Having 4096 feature maps in the deeper layers seemed unnecessary. 
  
#### Training and Testing
For training I gathered additional data using the CARLA simulator. In total the network was trained on 6400 images gathered from both towns and all 14 weather conditions. Images were preprocessed with histogram equalization. Denoising was also implemented by later removed due to its exponential time-complexity. Because not all classes were represented equally, loss was calculated using a weighted cross entropy function. Adam optimizer was used for backpropagation. During my research I learned that stochastic gradient decent tends to perform better over longer epochs. Due to time restrictions I decided to stick with Adam, which trains faster.

After each epoch the network was evaluated on 500 unseen images. This test dataset was specifically selected to contain hard-to-classify images, with small cars and dark/noisy conditions (i.e. hard rain during sunset). The weighted F-score, as described in the challenge, was calculated. Models with the best F-score were saved and retrained with lower learning rates to improve performance.
  
#### Prediction and FPS
Though the challenge involved classifying only vehicles and roads, I decided to segment as many classes as possible. As such my network is able to predict 10 classes. Of the original 13, 6 were combined: roads + road lines, buildings + walls, other + traffic signs. The intention behind this was an intuition that the network would better distinguish vehicles and roads if it had a greater classification capacity. This was backed by experimentation which showed improved F-scores (for vehicles and roads) with additional classes in prediction. However, this had a measurable negative effect on inference speed. The intention was to compensate for this loss by employing TensorRT. Unfortunately I experienced significant challenges in implementing TensorRT, and instead had to reduce the input image size to keep FPS above 10. The input image was trimmed 194 pixels from the top and 88 pixel from the bottom. This did not impact the final recall or precision scores, since the trimmed sections should not contain vehicles or roads. The image was then resized to approximately 2/3 the original resolution. This in particular brought the average f-score down from 0.919 to 0.915, which was my final submitted result.
  
  
## Results
In the end the network performed better than expected. I suspect this is due to the limited nature of data from the CARLA simulator, when compared to real-life images. The table below shows the F-scores from my submitted network, on my own test data:

| Class | F1 Score | F-beta Score |
|:---:|:---:|:---:|
|Background | 0.9549 | n/a
| Vehicles | 0.7994 | 0.8617
| Roads + Road lines | 0.9924 | 0.9917
| Fences | 0.7919 | n/a |
| Pedestrians | 0.7320 | n/a |
| Poles | 0.7993 | n/a |
| Sidewalks |  0.9660 | n/a |
| Vegetation | 0.8505 | n/a |
| Builds + Walls | 0.9090 | n/a |
| Other + Traffic Signs | 0.7948 | n/a |


#### Accurate Predictions
The network performed well on most images:

![alt text][image2]
  
#### Failures
In order to effectively evaluate the network, I chose specific images that I expected the network to fail on. For example the first two images in the grouping below. These two images were the very first frames in a new CARLA episode. They are darker than usual because the scene’s lighting is not fully loaded when the frame was taken. I tested on these images in particular to gauge performance in dark scenes. However, the prediction failures in the third image was not expected, and quite interesting. It seems the pedestrian's guitar case is being classified as a car, while it overlaps with the road. Such situations are important to guide the network's training through points-of-failure.  
  
![alt text][image3]
  
  
## Enhancements
  
In order to meet FPS requirements I trimmed and resized the input image. This is not ideal and my preferred solution was to use TensorRT. Other methods such as separate client/server inference processes are less effective, though ideally one should use both techniques. I was able to setup the workspace to run TensorRT, but could not convert my model due to incompatible layers. In retrospect I should have tested this before fine-tuning my model, given limited time. My plan for the future is to run inference using TensorRT in C++, with multi-processing.
  
Another major area of improvement is the network design itself. My model is close to a basic implementation. Having gone through the process of reading articles on semantic segmentation, I have developed a solid understanding of the challenges involved. My intention is to continue down this path and implement some of the more complex models such as PSPNet and DeepLabv3. Additionally I want to make sure of temporal information available in videos. Information of predictions from one frame should be beneficial for predication probabilities in the next frame (assuming they are continuous).
  
