# **Traffic Sign Recognition** 

---

The goals / steps of this project are the following:
* Load the German Traffic Sign data set
* Explore, summarize and visualize the data set
* Design, train and test a Neural network model architecture
* Use the model to make predictions on new images (found on internet)
* Analyze the softmax probabilities of the new images
* Visualization the activations of convolution layers 

---

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

* The size of training set is 35288
* The size of the validation set is 3921
* The size of test set is 12630
* The shape of a traffic sign image (after resizing) is (32, 32, 1)
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes in training dataset.
<img src="/plots/train_ds_dist.png" />

Number of training examples before data augmentation :  39209

Max examples for a class   :  2250

Min examples for a class   :  210

#### 3. Data Augmentation

The first issue that was addressed was dataset class skewness, this was done using augmenting the training dataset with random transformation. Random transformations include X-shift, y-shift, sheer, rotation. A utils function named `balance_training_dataset()` was created to augment dataset.

After augmenting the dataset the class distribution now were:
<img src="/plots/train_ds_dist_after.png" />

Below are class disitribution statistics after augmentation:

Number of training examples after data augmentation :  96596

Number of examples added to training dataset        :  57387

Max examples for a class :  2250

Min examples for a class :  2240

#### 4. Data Preprocessing

ImageDataGenerator was used to preprocess batches of images and split the training dataset into training dataset and validation dataset. Preprocessing operations performed were
- RGB to grayscale conversion
- resizing images from various resoltions to image size (32, 32)
- Normalization
- Zero Mean and uniform variance
- Histogram Equalization

### Classification Model

#### 1. Model Architecture

My final model consisted of the following layers:

| Layer (type)        | Output Shape          | Param #   
|---                  |:---                   |:---
| Conv2D              | (None, 30, 30, 64)    | 640       
| BatchNomalization   | (None, 30, 30, 64)    | 256       
| Activation('relu')  | (None, 30, 30, 64)    | 0         
| Conv2D              | (None, 28, 28, 64)    | 36928     
| BatchNomalization   | (None, 28, 28, 64)    | 256       
| Activation('relu')  | (None, 28, 28, 64)    | 0         
| MaxPooling2D        | (None, 14, 14, 64)    | 0         
| Conv2D              | (None, 12, 12, 128)   | 73856     
| BatchNomalization   | (None, 12, 12, 128)   | 512       
| Activation('relu')  | (None, 12, 12, 128)   | 0         
| Conv2D              | (None, 10, 10, 128)   | 147584    
| BatchNomalization   | (None, 10, 10, 128)   | 512       
| Activation('relu')  | (None, 10, 10, 128)   | 0         
| MaxPooling2D        | (None, 5, 5, 128)     | 0         
| Flatten             | (None, 3200)          | 0         
| Dense               | (None, 356)           | 1139556   
| BatchNomalization   | (None, 356)           | 1424      
| Activation('relu')  | (None, 356)           | 0         
| Dropout             | (None, 356)           | 0         
| fc2 (Dense)         | (None, 356)           | 127092    
| BatchNomalization   | (None, 356)           | 1424      
| Activation('relu')  | (None, 356)           | 0         
| Dropout             | (None, 356)           | 0         
| fc3 (Dense)         | (None, 43)            | 15351     

Total params: 1,545,391
Trainable params: 1,543,199
Non-trainable params: 2,192
_________________________________________________________________

#### 3. Training methodology

Train-validation split was 70-30. Adam Optimizer with intitial learning rate 5e-3 was used, however the learning rate was scaled down by a factor of 10 every 5 epochs if the validation-accuracy did not increase by 0.05%. The model was trained for 50 epochs with an early stop on training if the validation-accuracy did not increase by 0.05% for 15 epochs. Intermediate Fully Connected Layer were regularized using dropout with a dropout factor of 50% and final layer was regularized using l2 loss. All activations from Convolution layers and fully-connected-layers were batch-normalized as well. Batch-size used was 64, and the model was trained on Nvidia Gt 1050Ti 4GB gpu for about 25 minutes.

#### 4. Model performance

* validation set accuracy : 0.99 
* validation set accuracy : 0.96 
* test set accuracy : 0.95
* test set precision : 0.95
* test set recall : 0.91

#### 5. Training tradeoffs
- Initial model was based on LeNet architecture was performed fairly well with a test set accuracy of 0.90
- Later a higher capicity model was build to handle the augmented dataset
- Model performance improved when grayscale images were used instead of rgb images
- Batch Normalization significantly improved training speed

### Testing Model on New Images

60 traffic sign images from internet were collected and these images were saved in the directory `./data/examples/` in their appropriate class folders. Neural network model was trained on these example images, to check its perfomance. To see a detailed analysis of the Model prediction on random web images of traffic sign goto [notebook](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb). The analysis includes the Top-5 predictions as well as the softmax prediction probabilities of the new images as a bar chart.

#### Observations

 - A general issue that emerges is that model struggles to correctly classify images that have random noise in them. For e.g.: the watermarks on the images. or the viewing angle of the image is acute.

- Model finds it difficult to distinguish between similar sign

- Model not only learn the sign but also learns the shape of the sign board, this makes it difficult for the model to correctly classifly the same sign on a different shaped board.

### Visualizing the Neural Network

Finally, the activations of Conv1 of Block-1 (for any 2 random images in the test dataset) were visualized. Conv1 layers seems to extract edge information from images at different angles.

### Future Work

- Augmenting the dataset with random zooms and random brighness change
- Experiment with a model containing skip connections
