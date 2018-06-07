# Lyft's Perception Challenge

### Table of Contents
  1) Overview
  2) Related Work
  3) Implementation
  4) Results
  5) Shortcomings and Enhancements

[image1]: ./images/network.png "Network architecture"
[image2]: ./images/good.png "Good prediction results"
[image3]: ./images/bad.png "Bad prediction results"

## Overview
This challenge involved producing pixel-by-pixel annotations of vehicles and roads in images from the [CARLA simulator](www.carla.org), referred to as semantic segmentation. A modified Fully Convolutional Network (FCN) was constructed using TensorFlow in Python. Dilated convolutions were used in place of deeper max-pooling to increase the network’s receptive field, while maintaining granularity. A skip connection was added to the network to preserve finer details lost during earlier max-pooling. Loss was caluated using weighted cross entropy, optimized with Adam. 6400 images were used for training (both towns, all weather conditions), and 500 specifically selected images (difficult scenes) for testing. The resulting architecture ranked 13th in the competition with a final score of 91.49.

## Related Work
This was my first time implementing semantic segmentation, as such I closely followed the designs outlined in the following papers:
  
1)	[Fully Convolutional Networks for Semantic Segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)  
As a starting point I adopted their approach, using a pre-trained VGG with convolutional replacement and up-sampled skip connections. The results were impressive but coarse due to loss of finer details during max-pooling.
  
2)	[Multi-Scale Context Aggregation by Dilated Convolutions]( https://arxiv.org/abs/1511.07122)  
In order to produce more granular segmentations I redesigned my network with dilated convolutions. The last max-pooling layer was replaced with multiple dilated convolutions, increasing the networks receptive field while preserving more details. Unlike the authors' design, I kept one skip connection (from the third convolutional stack) to improve precision and recall. This skip connection was scaled by a factor of 0.01 to give preference to outputs from the deeper layers, and avoid larger weight updates to the shallow layers.


## Implementation
#### Network Architecture
The final network design contained 19 layers (13 regular convolutions, 4 dilated convolutions, and 2 transposed convolutions), illustrated in the image below. Note the output node contains the last transposed convolution.
   
![alt text][image1]
  
I initialized the first 13 layers with [pre-trained weights](https://drive.google.com/open?id=0Bx9YaGcDPu3XR0d4cXVSWmtVdEE). The remaining layers were implemented from scratch, with reduced depth (from 4096 to 512) for faster inference. CARLA data contains a limited number of skins and textures. Having 4096 feature maps in the deeper layers seemed unnecessary. 
  
#### Training and Testing
I obtained additional training data using the CARLA simulator. In total the network was trained on 6400 images gathered from both towns and all 14 weather conditions. Images were preprocessed with histogram equalization and randomly flipped (horizontal). Denoising was implemented but later removed due to its exponential time-complexity and limited benefit. L2 regularization was used to mitigate overfitting. Since classes were not represented equally, loss was calculated using a weighted cross entropy function. This was combined with the loss from regularization and minimized through the Adam optimizer.

After each epoch the network was evaluated on 500 unseen images. This test dataset was specifically selected to contain hard-to-classify images, with small cars and dark/noisy conditions (i.e. hard rain during sunset). The weighted F-score, as described in the challenge, was calculated. Models with the best F-score were saved and retrained with lower learning rates.
  
#### Prediction and FPS
Though the challenge involved classifying only vehicles and roads, I decided to segment as many classes as possible. As such my network is able to predict 10 classes. Of the original 13, 6 were combined to reduce memory requirements: roads + road lines, buildings + walls, other + traffic signs. The intention behind this was an intuition that the network would better distinguish vehicles and roads if it had a greater classification capacity. This was backed by experimentation which showed improved F-scores (for vehicles and roads) with additional classes in prediction. However, it had a measurable negative effect on inference speed. The intention was to compensate for this loss by employing TensorRT. Unfortunately I experienced challenges in implementing TensorRT, and at the last minute reduced the input image size to keep FPS above 10. The input image was trimmed 194 pixels from the top and 88 pixel from the bottom. This had no effect on precision and recall scores, since the trimmed sections did not contain vehicles or roads. The image was then resized to approximately 2/3 the original resolution. This in particular brought the average f-score down from 0.919 to 0.915, which was my final submitted result.
  
  
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
In order to effectively evaluate the network, I chose specific images that I expected the network to fail on. For example the first two images in the grouping below. These were the very first frames in a new CARLA episode. They were darker than usual because the scene’s lighting was not fully loaded when the frame was taken. I tested on such images in particular to gauge performance in difficult scenes. The prediction failures in the third image was not expected, and quite interesting. The pedestrian's guitar case was classified as a car, only while it overlaped with the road. Such situations are important to guide the network's training through points-of-failure.  
  
![alt text][image3]
  
  
## Shortcomings and Enhancements
  
During my research I discovered that stochastic gradient descent with moment can outperform Adam in the long run. Given more time I recommend using SGD + moment with simulated annealing, over Adam optimizer.
   
In order to meet FPS requirements I trimmed and resized the input image as a last-minute hack. This is not ideal and my preferred solution is to use TensorRT. Other methods such as separate client/server inference processes are less effective, though ideally one should use both techniques. I was able to setup the workspace to run TensorRT, but could not convert my model due to incompatible layers. In retrospect I should have tested this before fine-tuning my model. My plan for the future is to run inference using TensorRT in C++, with multi-processing.
  
Another major area of improvement is the network design itself. My model is close to a basic implementation. Having gone through this challenge, I have developed a solid understanding of the task. In the future I intend to implement advanced models such as PSPNet and DeepLabv3. Additionally I want use the temporal data available in videos. Information on predictions from one frame should inform predication probabilities in the next (assuming they are continuous). I have decided to make this the focus of my master's project.
  
