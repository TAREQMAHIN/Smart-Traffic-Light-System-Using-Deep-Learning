# Smart-Traffic-Light-System-Using-Deep-Learning

**Integration of a vehicle detection system in a traffic light camera, that could easily track the number of useful things simultaneously.**

● How many vehicles are present at the traffic junction during the day? 

● What time does the traffic build-up? 

● What kind of vehicles are traversing the junction (heavy vehicles, cars, etc.)?

● Is there a way to optimize the traffic and distribute it through a different street?


## The implementation of the project can be done in Three phases. All pashes are described bellow:

# DATASET: 

## GTI DATA:

GTI Dataset: http://www.gti.ssr.upm.es/data/Vehicle_database.html
Get Data: http://www.gti.ssr.upm.es/data/index.html

The Image Processing Group is currently researching on the vision-based vehicle classification task. In order to evaluate methods, we have created a new Database of images that we have extracted from our video sequences (acquired with a forward looking camera mounted on a vehicle). The database comprises 3425 images of vehicle rears taken from different points of view, and 3900 images extracted from road sequences not containing vehicles. Images are selected to maximize the representativity of the vehicle class, which involves a naturally high variability. In our opinion one important feature affecting the appearance of the vehicle rear is the position of the vehicle relative to the camera. Therefore, the database separates images in four different regions according to the pose: middle/close range in front of the camera, middle/close range in the left, close/middle range in the right, and far range. In addition, the images are extracted in such a way that they do not perfectly fit the contour of the vehicle in order to make the classifier more robust to offsets in the hypothesis generation stage. Instead, some images contain the vehicle loosely (some background is also included in the image), while others only contain the vehicle partially. Several instances of a same vehicle are included with different bounding hypotheses. The images have 64x64 and are cropped from sequences of 360x256 pixels recorded in highways of Madrid, Brussels and Turin.

## KITTI DATASET: 

KITTI Dataset: http://www.cvlibs.net/datasets/kitti/
Get Data: http://www.cvlibs.net/datasets/kitti/eval_3dobject.php

Kitti contains a suite of vision tasks built using an autonomous driving platform. The full benchmark contains many tasks such as stereo, optical flow, visual odometry, etc. This dataset contains the object detection dataset, including the monocular images and bounding boxes. The dataset contains 7481 training images annotated with 3D bounding boxes. A full description of the annotations can be found in the readme of the object development kit readme on the Kitti homepage.



# In the 1st phase we have to divide traffic junction into three different phases:

> High traffic zone

> Medium traffic zone

> Low traffic zone 

We will also conqure the traffic vehicles into different segments like heavy vehicles,medium heavy vehicles and small vehicles. 

We will use Semantic segmentation, Histrogram of oriented gradient , thresh holding and egde detection for the segmentation of the traffic vehicles into different zones. 

>**Semantic Segmentation:** Semantic segmentation refers to the process of linking each pixel in an image to a class label. These labels could include a person, car, flower, piece of furniture, etc.  of a Deep Convolutional Network for Semantic Image Segmentation.It classify all the images into different pixel and classes. 

> **Histogram of oriented gradient(HOG):** It uses histogram of group of pixels based on "Gray Level". 

> **Thresh Holding:** Threshholding means devide an image into foreground and background. 

> **Edge Detection:** Identify Sharp changes and discontinueties in brightness. 

![](Trafic%20Segmentation/Capture2.png)

![Capture2](https://user-images.githubusercontent.com/49672241/96851652-e0317980-1475-11eb-863b-a572231595d7.png)



## Frameworks and Packages:

-Python 3
-TensorFlow
-NumPy
-SciPy



# PHASE 2: 

In this phase of the project I would like to implement a supervised classification algorithm for the purpose of seperating the trafffic vehicles. I would use the SUPPORT VECTOR MACHINE (SVM) for the classificication of the traffic vehicles image. I would also use the Histogram of Oriented Gradient(HOG) for feature extarcting from data.

After importing all the libreries we will visualize the vehicle and non vehicle images by the following code:
![Capture](https://user-images.githubusercontent.com/49672241/96862156-ba5ea180-1482-11eb-80b1-7aa8265a62b5.PNG)


![car](https://user-images.githubusercontent.com/49672241/96862339-f5f96b80-1482-11eb-8140-1a0d4fe50ca6.png)
![no car](https://user-images.githubusercontent.com/49672241/96862508-30fb9f00-1483-11eb-98c3-d800e26daaf4.png)





The next part is Feature Extraction. I have used the histogram of Oriented Gradient (HOG) for extarcting the  feature form the image set data. The code of the feature extraction is given bellow with it's corresponding result .
![Capture2](https://user-images.githubusercontent.com/49672241/96863231-2c83b600-1484-11eb-97b0-fb860c4d7fee.PNG)

![Feature1](https://user-images.githubusercontent.com/49672241/96863418-78cef600-1484-11eb-8243-d6ae55e9793e.png)
![Feature2](https://user-images.githubusercontent.com/49672241/96863441-7ff60400-1484-11eb-9439-15da989cfe47.png)

Now we are ready for HOG Testing of the dataset.
![Capture3](https://user-images.githubusercontent.com/49672241/96863650-d105f800-1484-11eb-9548-3ab5522aa0e7.PNG)


I decided to use Support Vector Machines because they have good compatibility with HOG. Now in SVM we have SVC(Support Vector Classifier) and here also I have a choice with various kernels and different C and gamma values.I trained my classifier on Linear kernel. The linear kernel took around 1.8 seconds to train with a test accuracy of 98.7%. I decided to use LinearSVC with default parameters solely because it was taking less time to run and it was more accurate.
![Capture4](https://user-images.githubusercontent.com/49672241/96863876-2b9f5400-1485-11eb-942f-7bf154870359.PNG)


![Capture1](https://user-images.githubusercontent.com/49672241/96866954-b2563000-1489-11eb-9ba9-beb8f1015a79.png)



**The linear kernel took around 1.8 seconds to train with a test accuracy of 98.7%. Input image length is 64x64**
![Capture5](https://user-images.githubusercontent.com/49672241/96864147-9b154380-1485-11eb-9ad1-5f7e33e1c002.PNG)

We can also illustarte the output bellow: 

![30512919-72a57046-9b02-11e7-8ca8-c3e4bd993497](https://user-images.githubusercontent.com/49672241/97027567-89609880-1578-11eb-9a0a-dd0ca9587ca6.gif)




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



So, by this algorithm we can easily optimize the signal configuration of high-level traffic zones to bring them to either medium-level or low-level traffic zones. 


### FINAL OBSERVATION:
We can have all the expected results as the project demands. By the integration of the project we
can easily build a smart traffic management system. Proper deployment would give us the following
outputs:

➢ We can easily count the vehicles at the traffic junction on real time and also can separate the
vehicles into different zones.

➢ By the help of the real time data we can also monitor the traffic congestion time in a particular
area at a particular time.

➢We can also cluster the traffic vehicles and classify them to different junctions to avoid congestion
on the road.

➢ Finally by implementing the deep learning model in this project involves in traffic signal
modification with the help of artificial intelligence to optimize traffic and distribute it through a
different street. 




