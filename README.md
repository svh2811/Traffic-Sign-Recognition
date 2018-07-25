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
* Visualization the activations of convolution layers 

--

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

[Training Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-training-dataset)

[Validation Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-validation-dataset)

[Testing Dataset Class Distribution](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Frequency-of-Class-Examples-in-test-dataset)

### Data Preprocessing

While creating tensorflow batch dataset every image in the german traffic sign dataset was resized to dimension (32, 32, 3) and then this image was normalized. [normalization details](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Normalization)

#### Model Architecture 

My final model consisted of the following layers:

| Layer | Description | 
|:--------------------:|:--------------------:| 
| Input | (?, 32, 32, 3) RGB image | 
| Convolution 5x5 | 1x1 stride, valid padding, Out: (?, 28, 28, 6) |
| RELU |	Out: (?, 28, 28, 6) |
| Max pooling | 2x2 stride,  Out: (?, 14, 14, 6) |
| Dropout | Out: (?, 14, 14, 6) |
| Convolution 5x5 | 1x1 stride, valid padding, Out: (?, 10, 10, 16)|
| RELU |	Out: (?, 10, 10, 16) |
| Max pooling | 2x2 stride,  Out: (?, 5, 5, 16) |
| Dropout | Out: (?, 5, 5, 16) |
| Flatten | Out: (?, 400) | 
| Fully connected | Out: (?, 120) |
| Fully connected | Out: (?, 84) |
| Softmax | Out: (?, 43) |

#### 3. Training methodology

I used Adam Optimizer with learning rate 3e-4. The model was trained for 50 epochs with dataset-batch-size: 32.

#### 4. Model evaluation

My final model results were:
* validation set accuracy of 0.96 
* test set accuracy of 0.94

### Testing Model on New Images

I collected 18 traffic sign images from internet and these images are saved in the directory `./examples` and the filename have the format `class-number_index.image-extension`.

I tested my neural network model on these example images.

Top 5 predictions with the corresponding class labels are displayed in this table [this table](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Top-K-Prediction-Table)

Next [confusion matrix](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Confusion-Matrix-Plot) for these predictions were plotted.

My model predicted correct images with an accuracy of 0.72 where 13/18 images were correctly classified, with a 
mean class precison of 0.19
mean class recall of 0.20
mean class specificity of 0.99
and F1 Score is 0.19

precision and recall score are relatively low, one major reason that explains these scores is there are only 18 examples and some class images are not included at all.

A general issue that emerges is that model struggles to correctly classify images that have random noise in them. For e.g.: the watermarks on the images. or the vewing angle of the image is acute.

### [Visualizing the Neural Network](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb#Step-5:-Visualize-the-Neural-Network's-State-with-Test-Images)

Finally, the activations of Conv1 and Conv2 layers (for any 3 random images in the test dataset) were visualized. Conv1 layers seems to extract edge information from images at different angles. Conv2 visualizations are difficult to interpret, it seems to extract primary image artifacts. 

### Future Work

Fromm the dataset class distribution, we see that the dataset class distribution is skewed. To fix this issue we can augmennt the data to balance the class distributions.
