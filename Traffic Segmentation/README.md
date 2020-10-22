# In the 1st phase we have to devide traffic juction into three different phases:

> High traffic zone

> Medium traffic zone

> Low traffic zone 

We will also conqure the traffic vehicles into different segments like heavy vehicles,medium heavy vehicles and small vehicles. 

We will use Semantic segmentation, Histrogram of oriented gradient , thresh holding and egde detection for the segmentation of the traffic vehicles into different zones. 

>**Sementic Segmentation:** Semantic segmentation refers to the process of linking each pixel in an image to a class label. These labels could include a person, car, flower, piece of furniture, etc.  of a Deep Convolutional Network for Semantic Image Segmentation.It classify all the images into different pixel and classes. 

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
