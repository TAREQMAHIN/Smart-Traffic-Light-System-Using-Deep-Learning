# PHASE 3 

Optimization of traffic vehicles into differernt zones and building a traffic sign recognition classifier with Convolutional Neural Network by the help of the Tensorlow2.0 library.

**The goal of this phase is to build a CNN in Tensorflow to classify traffic sigh images from GTI and KITTI dataset**

## MODEL EVALUATION: 

The model uses to build the tarffic conjunction more accurate without any human interation. The Images that are feed as the input of the neural network will be classified into differnt zones. The data will be labeled or unlabeled data that can be use Single Shot Detector (SSD) model in deep learning. The bounding box techniques will be used to extarct the appropiate feature from the input data. The tarffic density calculation will be done by cognitive intelligence to build a traffic light optimization. The multi-threaded algorithm  adaptive tarffic light for better conjetion optimization. 


![Capture](https://user-images.githubusercontent.com/49672241/96913378-2829be00-14c1-11eb-9163-ec63acd2c71e.png)

The pickled data is a dictionary with 4 key/value pairs:

> 'features' is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
> 'labels' is a 2D array containing the label/class id of the traffic sign. The file signnames.csv contains id -> name mappings for each id.
> 'sizes' is a list containing tuples, (width, height) representing the the original width and height the image.
> 'coords' is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. 

![Capture1](https://user-images.githubusercontent.com/49672241/96913883-dcc3df80-14c1-11eb-8c2d-7511d34413d3.PNG)


Now, I will illustrate the training dataset into the matplotlib librery and show thr random sample from the training set. By visualizing of the data we can have the following result. 

![Capture2](https://user-images.githubusercontent.com/49672241/96914320-75f2f600-14c2-11eb-99e4-4479ec459223.PNG)


The dataset also visualize the classes of each training and testing set. From this plot we notice that there's a strong imbalance among the classes. Indeed, some classes are relatively over-represented, while some others are much less common. However, we see that the data distribution is almost the same between training and testing set.

![Capture3](https://user-images.githubusercontent.com/49672241/96914806-1812de00-14c3-11eb-967a-b90269d82b5d.PNG)

***Now if we run the model in the CNN network we can easily get the acuracy of the testing model.The testing accuacy is about 95%. on the other hand the training set will classify into maximum 30 epoch and the accuracy predicted by the model is around 98%.***

![Capture5](https://user-images.githubusercontent.com/49672241/96915313-ba32c600-14c3-11eb-8313-bd4fac983a36.PNG)
![Capture4](https://user-images.githubusercontent.com/49672241/96915334-bdc64d00-14c3-11eb-9efd-515389733638.PNG)


At last if the model is implemented by real time data it also shows a high level accuracy of around 80%.


![Capture6](https://user-images.githubusercontent.com/49672241/96915875-6a083380-14c4-11eb-9132-6f69208d2ec7.PNG)

So, by this algorithm we can easily optimize the signal configuration of high-level traffic zones to bring them to either medium-level or low-level traffic zones. 




