# PHASE2: 

In this phase of the project we will implement a supervised classification algorithm for the purpose of seperating the trafffic vehicles. We will use the SUPPORT VECTOR MACHINE (SVM) for the classificication of the traffic vehicles image. We would also use the Histogram of augmented gradient(HOG) for feature extarcting from data.

After importing all the libreries we will visualize the vehicle and non vehicle images by the following code:
![Capture](https://user-images.githubusercontent.com/49672241/96862156-ba5ea180-1482-11eb-80b1-7aa8265a62b5.PNG)


![car](https://user-images.githubusercontent.com/49672241/96862339-f5f96b80-1482-11eb-8140-1a0d4fe50ca6.png)
![no car](https://user-images.githubusercontent.com/49672241/96862508-30fb9f00-1483-11eb-98c3-d800e26daaf4.png)





The next part is Feature extraction. I have used the histogram of oriented gradient (HOG) for extarcting the  feature form the image set data. The code of the feature extraction is given bellow with its corresponding result .
![Capture2](https://user-images.githubusercontent.com/49672241/96863231-2c83b600-1484-11eb-97b0-fb860c4d7fee.PNG)

![Feature1](https://user-images.githubusercontent.com/49672241/96863418-78cef600-1484-11eb-8243-d6ae55e9793e.png)
![Feature2](https://user-images.githubusercontent.com/49672241/96863441-7ff60400-1484-11eb-9439-15da989cfe47.png)

Now we are ragy for HOG Testing of the dataset.
![Capture3](https://user-images.githubusercontent.com/49672241/96863650-d105f800-1484-11eb-9548-3ab5522aa0e7.PNG)


I decided to use Support Vector Machines because they have good compatibility with HOG. Now in SVM we have SVC(Support Vector Classifier) and here also we have a choice with various kernels and different C and gamma values.I trained my classifier on Linear kernel. The linear kernel took around 1.8 seconds to train with a test accuracy of 98.7%. I decided to use LinearSVC with default parameters solely because it was taking less time to run and it was more accurate.
![Capture4](https://user-images.githubusercontent.com/49672241/96863876-2b9f5400-1485-11eb-942f-7bf154870359.PNG)



**The linear kernel took around 1.8 seconds to train with a test accuracy of 98.7%. Input image length is 64x64**
![Capture5](https://user-images.githubusercontent.com/49672241/96864147-9b154380-1485-11eb-9ad1-5f7e33e1c002.PNG)




