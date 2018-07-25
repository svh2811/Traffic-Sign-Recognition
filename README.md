# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the German Traffic Sign data set
* Explore, summarize and visualize the data set
* Design, train and test a Neural network model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualza the activations of convolution layers 


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Code Workflow

Here is a link to my [project code](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

* The size of training set is 35288
* The size of the validation set is 3921
* The size of test set is 12630
* The shape of a traffic sign image (after resizing) is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes. [project code](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb)

![Training Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-training-dataset)

![Validation Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-validation-dataset)

![Testing Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-test-dataset)

### Data Preprocessing

While creating tensorflow batch dataset every image in the german traffic sign dataset was resized to dimension (32, 32, 3) and then this image was normalized. [normalization details](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Normalization)

#### Model Architecture 

My final model consisted of the following layers:

| Layer         		     | Description                                    | 
|:--------------------:|:----------------------------------------------:| 
| Input         		     | (?, 32, 32, 3) RGB image   							             | 
| Convolution 5x5     	| 1x1 stride, valid padding, Out: (?, 28, 28, 6) |
| RELU					            |	Out: (?, 28, 28, 6)                            |
| Max pooling	      	  | 2x2 stride,  Out: (?, 14, 14, 6) 				          |
| Dropout              | Out: (?, 14, 14, 6)                            |
| Convolution 5xs     	| 1x1 stride, valid padding, Out: (?, 10, 10, 16)|
| RELU					            |	Out: (?, 10, 10, 16)                           |
| Max pooling	      	  | 2x2 stride,  Out: (?, 5, 5, 16) 				           |
| Dropout              | Out: (?, 5, 5, 16)                             |
| Flatten              | Out: (?, 400)                                  | 
| Fully connected		    | Out: (?, 120)								                          |
| Fully connected		    | Out: (?, 84) 								                          |
| Softmax				          | Out: (?, 43) 								                          |

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


