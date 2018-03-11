## Reference
[Introduction to CNN Keras - 0.997 (top 6%)](https://www.kaggle.com/yassineghouzam/introduction-to-cnn-keras-0-997-top-6)

## 1.文中好的句子的摘录  

1. 
> We perform a grayscale normalization to reduce the effect of illumination's differences.   

> Moreover the CNN converg faster on [0..1] data than on [0..255].

2. 
> Since we have 42 000 training images of balanced labels (see 2.1 Load data), a random split of the train set doesn't cause some labels to be over represented in the validation set. Be carefull with some unbalanced dataset a simple random split could cause inaccurate evaluation during the validation. To avoid that, you could use `stratify` option in train_test_split function (Only for >=0.17 sklearn versions).

3. 
> Filters can be seen as a transformation of the image.   

> The CNN can isolate features that are useful everywhere from these transformed images (feature maps).  

> These are used to reduce computational cost, and to some extent also reduce overfitting. 

4. 
> Combining convolutional and pooling layers, CNN are able to combine local features and learn more global features of the image.

5. 
> Dropout improves generalization and reduces the overfitting.

6. 
> The Flatten layer is use to convert the final feature maps into a **one single 1D vector**. This flattening step is needed so that you can make use of fully connected layers after some convolutional/maxpool layers. It combines all the found local features of the previous convolutional layers.

7.
> The most important function is the optimizer. This function will iteratively improve parameters (filters kernel values, weights and bias of neurons ...) in order to minimise the loss.  

> We could also have used Stochastic Gradient Descent ('sgd') optimizer, but it is slower than RMSprop.

8.
> In order to make the optimizer converge faster and closest to the global minimum of the loss function, I used an annealing method of the learning rate (LR).  

> Its better to have a decreasing learning rate during the training to reach efficiently the global minimum of the loss function.  

> To keep the advantage of the fast computation time with a high LR, I decreased the LR dynamically every X steps (epochs) depending if it is necessary (when accuracy is not improved).  

> With the ReduceLROnPlateau function from Keras.callbacks, I choose to reduce the LR by half if the accuracy is not improved after 3 epochs.

```python
# Set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
```

9. 
Data Augmentation
> In order to **avoid overfitting** problem, we need to expand artificially our handwritten digit dataset. We can make your existing dataset even larger. **The idea is to alter the training data with small transformations to reproduce the variations occuring when someone is writing a digit**.  

> For example, the number is not centered/The scale is not the same (some who write with big/small numbers)/The image is rotated...  

> **Approaches that alter the training data in ways that change the array representation while keeping the label the same are known as data augmentation techniques**. Some popular augmentations people use are **grayscales, horizontal flips(翻转), vertical flips, random crops, color jitters, translations, rotations, and much more**.  

> By applying just a couple of these transformations to our training data, we can easily double or triple the number of training examples and create a very robust model.  

> **The improvement is important**:  
> + Without data augmentation I obtained an accuracy of 98.114%
> + With data augmentation I achieved 99.67% of accuracy

> For the data augmentation, I choosed to :
> + Randomly rotate some training images by 10 degrees
> + Randomly Zoom by 10% some training images
> + Randomly shift images horizontally by 10% of the width
> + Randomly shift images vertically by 10% of the height
> + I did not apply a vertical_flip nor horizontal_flip since it could have lead to misclassify symetrical numbers such as 6 and 9.


## 2.ReduceLROnPlateau
从下面的结果中我们可以看出使用ReduceLROnPlateau还是很有效的
```
Epoch 26/100
 - 28s - loss: 0.0646 - acc: 0.9789 - val_loss: 0.0304 - val_acc: 0.9913

Epoch 00026: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 27/100
 - 28s - loss: 0.0616 - acc: 0.9806 - val_loss: 0.0290 - val_acc: 0.9907
Epoch 28/100
 - 28s - loss: 0.0583 - acc: 0.9807 - val_loss: 0.0297 - val_acc: 0.9906
Epoch 29/100
 - 28s - loss: 0.0556 - acc: 0.9822 - val_loss: 0.0285 - val_acc: 0.9914

Epoch 00029: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
Epoch 30/100
 - 28s - loss: 0.0529 - acc: 0.9833 - val_loss: 0.0283 - val_acc: 0.9921
Epoch 31/100
 - 28s - loss: 0.0534 - acc: 0.9837 - val_loss: 0.0288 - val_acc: 0.9916
Epoch 32/100
 - 28s - loss: 0.0532 - acc: 0.9839 - val_loss: 0.0285 - val_acc: 0.9917
Epoch 33/100
 - 28s - loss: 0.0540 - acc: 0.9830 - val_loss: 0.0276 - val_acc: 0.9921

Epoch 00033: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
Epoch 34/100
 - 28s - loss: 0.0515 - acc: 0.9835 - val_loss: 0.0273 - val_acc: 0.9922
Epoch 35/100
 - 28s - loss: 0.0556 - acc: 0.9836 - val_loss: 0.0271 - val_acc: 0.9924
Epoch 36/100
 - 28s - loss: 0.0504 - acc: 0.9841 - val_loss: 0.0272 - val_acc: 0.9923
Epoch 37/100
 - 28s - loss: 0.0516 - acc: 0.9841 - val_loss: 0.0269 - val_acc: 0.9925
Epoch 38/100
 - 28s - loss: 0.0486 - acc: 0.9848 - val_loss: 0.0266 - val_acc: 0.9921

Epoch 00038: ReduceLROnPlateau reducing learning rate to 6.25000029685907e-05.
```

