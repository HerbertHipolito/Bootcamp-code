# Classification of lung nodules

Lung cancer is one of the most common cancers worldwide. Around 2.2 million new cases of lung cancer were registered just in 2020. 
Therefore, it is essential to create/develop tools to diagnose that sort of cancer as early as possible in order to increase the chances of survival. 
In this project, I used Convolutional Neural Network (CNN) and feature extraction to classify the lung cancer nodule between benign and malignant. 
It is a fundamental task as malignant nodules are larger, and more dangerous and tend to spread to other organs and damage them. The image below shows 4 nodules:

![nodulesgit](https://user-images.githubusercontent.com/94997683/213807109-b0e6f30a-2f1c-451c-a1c2-cfff70a57e87.png)

The top two images contain benign nodules and the bottom two images contain malignant nodules. All nodules in the picture are centralized.

As you probably noticed, the pictures have a lot of noise which can confuse the models, reducing our results.
Then segmentation has been made by applying the Flood Fill algorithm before using the CNN and feature extraction. Beyond noise removal, the images have been enlarged to
highlight the nodule traits.

![noduleseg](https://user-images.githubusercontent.com/94997683/213810416-f65292a0-6cd2-4aee-859a-d0dae51bfde6.png)

The image above contains the same nodules from image 1 after using flood fill and being enlarged. Now, the nodules are isolated and super larger. it is important
to say that the use of flood fill was possible once the nodules are centralized at 100x100 px. However, due the shape of same nodules, this segmentation fails 
as the seed is set outside the nodule, in a few images.

## Results

### Feature extraction

3 pipelines were created: 1) basic features such as area, perimeter, convex area, diameter and shannon entropy 2) Local Binary Patterns 3) First and second pipelines together.
For each pipeline, I used KNN, Naive Bayes, MLP, SVM, SGDC and Random forest.

The following table shows the result of third pipeline that obtained best results:

![table3](https://user-images.githubusercontent.com/94997683/213814759-8cee16af-3951-474e-bb5e-1b318e8472a5.png)

The results of other pipelines can be checked out in the file: Classification-of-lung-nodules/notebooks/features-classification.ipynb.

### CNN

Three architectures were set (each simulation represents a architectures) with kfolder k=5. 
You can verify all details regarding the architectures in the file located: Classification-of-lung-nodules/src/architectures.py

table 1) Average of the 5 folds.
table 2) Standard deviation of the 5 folds.

![tablefinalcnn](https://user-images.githubusercontent.com/94997683/213815454-3cfba1c3-2ba8-4630-9195-4fb018a04fe6.png)

The results, confusion matrix, graphs of simulations can be checked out in the file: Classification-of-lung-nodules/notebooks/CNN-classification-results.ipynb


