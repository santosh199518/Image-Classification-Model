# int-247-km007-ca-2-santosh199518
A project on Image Classification where we can classify the image into different classes based upon different attributes present on given dataset.

### It consists of three files:
- Segmentation_train.csv: The file containing the training data for classification.
- Segmentation_test.csv:  The File containing the testting data for classification.
- Segmentation.names:     The file contating names of attributes present in the data.\
**These data has been taken from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/image+segmentation#:~:text=UCI%20Machine%20Learning%20Repository%3A%20Image%20Segmentation%20Data%20Set&text=Data%20Set%20Information%3A,a%20classification%20for%20every%20pixel.).**


# Image Segmentation:
In digital image processing and computer vision, image segmentation is the process of partitioning a digital image into multiple segments (sets of pixels, also known as image objects). The goal of segmentation is to simplify and/or change the representation of an image into something that is more meaningful and easier to analyze. Image segmentation is typically used to locate objects and boundaries (lines, curves, etc.) in images. More precisely, image segmentation is the process of assigning a label to every pixel in an image such that pixels with the same label share certain characteristics.

The result of image segmentation is a set of segments that collectively cover the entire image, or a set of contours extracted from the image (see edge detection). Each of the pixels in a region are similar with respect to some characteristic or computed property, such as color, intensity, or texture.

# Some information about Dataset:
# Attribute Description:
-  region-centroid-col:  the column of the center pixel of the region.
- region-centroid-row:  the row of the center pixel of the region.
- region-pixel-count:  the number of pixels in a region = 9.
- short-line-density-5:  the results of a line extractoin algorithm that counts how many lines of length 5 (any orientation) with low contrast, less than or equal to 5, go through the region.
- short-line-density-2:  same as short-line-density-5 but counts lines
         of high contrast, greater than 5.
- vedge-mean:  measure the contrast of horizontally adjacent pixels in the region.  There are 6, the mean and standard deviation are given.  This attribute is used as a vertical edge detector.
- vegde-sd:  (see 6)
- hedge-mean:  measures the contrast of vertically adjacent pixels. Used for horizontal line detection. 
- hedge-sd: (see 8).
- intensity-mean:  the average over the region of (R + G + B)/3
- rawred-mean: the average over the region of the R value.
- rawblue-mean: the average over the region of the B value.
- rawgreen-mean: the average over the region of the G value.
- exred-mean: measure the excess red:  (2R - (G + B))
- exblue-mean: measure the excess blue:  (2B - (G + R))
- exgreen-mean: measure the excess green:  (2G - (R + B))
- value-mean:  3-d nonlinear transformation of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals of Interactive Computer Graphics)
- saturation-mean:  (see 17)
- hue-mean:  (see 17)
These are the real time attributes obtained from an image after segmentation. Now we need to classify the given images into different target classes which are:
# Classes:  
brickface, sky, foliage, cement, window, path, grass.
# Number of Attributes: 
19 continuous attributes with all numeric values.
# Number of Instances: 
- Training data: 210  
- Test data: 2100
# Missing Attribute Values:
No missing attributes are present in given dataset.

# Algorithms used:
- RandomForestClassifier:
Random forests or random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes (classification) or mean/average prediction (regression) of the individual trees. Random decision forests correct for decision trees' habit of overfitting to their training set. Random forests generally outperform decision trees, but their accuracy is lower than gradient boosted trees. However, data characteristics can affect their performance.
         
- Perceptron: 
A Perceptron is an algorithm used for supervised learning of binary classifiers. Binary classifiers decide whether an input, usually represented by a series of vectors, belongs to a specific class. In short, a perceptron is a single-layer neural network.
         
- Logistic Regression: 
Logistic regression is a supervised learning classification algorithm used to predict the probability of a target variable. The nature of target or dependent variable is dichotomous, which means there would be only two possible classes.
         
- Support Vector Classifier: 
In machine learning, support-vector machines (SVMs, also support-vector networks) are supervised learning models with associated learning algorithms that analyze data for classification and regression analysis. In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly mapping their inputs into high-dimensional feature spaces.
- Bagging Classifier:
A Bagging classifier is an ensemble meta-estimator that fits base classifiers each on random subsets of the original dataset and then aggregate their individual predictions (either by voting or by averaging) to form a final prediction. Such a meta-estimator can typically be used as a way to reduce the variance of a black-box estimator (e.g., a decision tree), by introducing randomization into its construction procedure and then making an ensemble out of it.

This algorithm encompasses several works from the literature. When random subsets of the dataset are drawn as random subsets of the samples, then this algorithm is known as Pasting. If samples are drawn with replacement, then the method is known as Bagging. When random subsets of the dataset are drawn as random subsets of the features, then the method is known as Random Subspaces. Finally, when base estimators are built on subsets of both samples and features, then the method is known as Random Patches.\
         
### I hope this repository was helpful for everyone to understand image classification from given dataset.
