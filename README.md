# **Traffic Sign Recognition** 

---

The goals / steps of this project are the following:
* Load the German Traffic Sign data set
* Explore, summarize and visualize the data set
* Design, train and test a Neural network model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualization the activations of convolution layers 

---

### Data Set Summary & Exploration

#### 1. Basic summary of the data set.

* The size of training set is 39209
  * Maximum examples for a class: 2250
  * Minimum examples for a class: 210

* The size of test set is 12630

* The shape of a traffic sign image (after resizing) is (32, 32, 1)

* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed across classes in training dataset.
<img src="/plots/train_ds_dist.png" />

Number of training examples before data augmentation :  39209

#### 3. Data Augmentation

The first issue that was addressed was dataset class skewness, this was done using augmenting the training dataset with random transformation. Random transformations include X-shift, y-shift, sheer, rotation. A utils function named `balance_training_dataset()` was created to augment dataset.

After augmenting the dataset the class distribution now were:
<img src="/plots/train_ds_dist_after.png" />

Below are class disitribution statistics after augmentation:

* Number of training examples after data augmentation :  96609
* Number of examples added to training dataset        :  57400
* Max examples for a class :  2250
* Min examples for a class :  2239

#### 4. Data Preprocessing

ImageDataGenerator was used to preprocess batches of images and split the training dataset into training dataset and validation dataset. Preprocessing operations performed were
- RGB to grayscale conversion
- resizing images from various resoltions to image size (32, 32)
- Normalization
- Zero Mean and uniform variance
- Histogram Equalization

Additionally, while training the network we randomly zoom the image.

The training-validation split is 0.30. The entire training dataset is divided into training dataset and validation dataset

|                    | Sample Count | Batch count |
|--                  |---           |:---         |
| training dataset   | 67639        | 133         |
| validation dataset | 28970        | 57          |
| test dataset       | 12630        | 25          |

### Classification Model

#### 1. Model Architecture

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
| Dense               | (None, 1067)          | 3415467   
| BatchNomalization   | (None, 1067)          | 4268
| Activation('relu')  | (None, 1067)          | 0         
| Dropout             | (None, 1067)          | 0         
| Dense               | (None, 357)           | 381276
| BatchNomalization   | (None, 357)           | 1424    
| Activation('relu')  | (None, 357)           | 0      
| Dropout             | (None, 357)           | 0         
| Dense               | (None, 43)            | 15394     

* Total params: 4,078,377
* Trainable params: 4,074,761
* Non-trainable params: 3,616
_________________________________________________________________

#### 2. Training methodology

Adam Optimizer with intitial learning rate 5e-3 was used, however the learning rate was scaled down by a factor of 10 every 3 epochs if the validation-accuracy did not increase by 0.05%. The model configured to be trained for a maximum of 150 epochs but with an early stop on training if the validation-accuracy did not increase by 0.05% for 15 epochs. Eventually the training terminated early after 24 epochs. All activations from Convolution layers and fully-connected-layers were batch-normalized as well. Intermediate Fully Connected Layer were regularized using dropout with a dropout factor of 50% and final layer was regularized using l2 loss. Batch-size used was 512, and the model was trained on Nvidia Gt 1050Ti 4GB gpu for about 24 minutes.

#### 3. Model performance

* training set accuracy   : 0.9687 
* validation set accuracy : 0.9623 
* test set accuracy : 0.9615
* test set precision : 0.9552
* test set recall : 0.9336
* test set f1-score : 0.9394

#### 4. Experiments
* Initial model was based on LeNet architecture which had a test set accuracy of 0.89
  * This model was underfitting 
  * Various experiments to prevent underfitting and even overfit the dataset were futile
  * Some of these experiments were:
    * training model for many epochs
    * training model with different batch sizes
    * limited regularization
* Later a higher capicity model was devised
   * This model was overfitting the dataset with validation set accuracy of 0.89
   * Next, the training dataset was augmented so that all classes have approximately same number of examples, the model now was generalizing well with a validation set accuracy of 0.92
   * To further improve the accuracy of the model, grayscale images were used instead of rgb images. This helped model to improve accuracy by 1.5% and more importantly it was able to better classify web examples. However the recall on test dataset  was 0.90.
* In a effort to improve recall on test dataset:
  * convolution layers with more kernel were used, specifically the number of kernals in convolution layers of second block was increased from 64 to 128.
  * In addition, Batch Normalization layer was added which significantly improved training speed
  * This steps helped to improve model's test set accuracy to 0.96, however the model was overfitting on training dataset and  recall on test dataset 0.91
  * To prevent overfitting for this model version, batch size was increased from 64 to 512, larger batch size aided the model to prevent overfitting on training dataset and improved generalization.
  * Recall on test dataset now increased from 0.91 to 0.9336

### Testing Model on New Images

84 traffic sign images from internet were collected and these images were saved in the directory `./data/examples/` in their appropriate class folders. Neural network model predicted the top 5 probable class labels for these example images. To see a detailed analysis of the Model prediction on random web images of traffic sign goto [notebook](https://github.com/svh2811/Traffic-Sign-Recognition/blob/master/Traffic_Sign_Classifier.ipynb). The analysis includes:
* Ground truth class label
* Image who's class label was predicted
* Most likely class label
* Top 5 most likely class labels
* Bar chart showing the probability values of all 43 class probablities.


### Visualizing the Neural Network

Finally, the activations of Conv1 of Block-1 (for any 2 random images in the test dataset) were visualized. Conv1 layers seems to extract edge information from images at different angles.

### Observations

* The model succesfully learns the pattern of the traffic images, this can be inferred from test set evaluation metric.
* This model can correctly classify images with random transformation. However the model would stuggle if learned features are absent in the target image. For e.g.:
  * Model not only learn the sign but also learns the shape of the sign board, this makes it difficult for the model to correctly classifly the same sign on a different shaped board.
  * Images where target sign is not of the scale as the images in training dataset.
  * Images with multiple signs

### Future Work

* Augmenting the dataset with random brightness change.
* Since the model is learning redundant features like, sign board shape, an additional preprocessing step that can be applied is center-crop the image.
* Experiment with a model containing skip connections.
