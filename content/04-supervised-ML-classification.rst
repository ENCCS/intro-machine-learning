Supervised ML (I): Classification
=================================


.. questions::

   - why 

.. objectives::

   - Explain 

.. instructor-note::

   - 40 min teaching
   - 40 min exercises


Classification is a supervised ML task where the model predicts discrete class labels based on input features. 
It involves training a model on labeled data so that it can assign new data to predefined categories or classes based on patterns learned from labeled training data.

In binary classification, models predict one of two classes, such as spam or not spam for emails. Multiclass classification extends this to multiple categories, like classifying images as cats, dogs, or birds.

Common algorithms for classification task include k-Nearest Neighbors (k-NN), logistic Regression, decision tree, random forest, naive Bayes, support vector machine (SVM), gradient boosting, and neural networks.

In this episode we will perform supervised classification tasks to categorize penguins into three species -- Adelie, Chinstrap, and Gentoo -- based on their physical measurements (flipper length, body mass, *etc.*). We will build and train multiple classifier models as mentioned above. Each model will be evaluated using appropriate performance metrics like accuracy, precision, recall, and F1 score. By comparing the results across models, we aim to identify which classifier model provides the most accurate and reliable classification for this task.

<center><img src="https://enccs.github.io/deep-learning-intro/_images/palmer_penguins.png" width="512"></center>



