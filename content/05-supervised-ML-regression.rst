Supervised Learning (II): Regression
====================================


.. questions::

   - why 

.. objectives::

   - Explain 

.. instructor-note::

   - 40 min teaching
   - 40 min exercises


Regression is a type of supervised ML task where the goal is to predict a continuous numerical value based on input features. Unlike classification, which assigns inputs to discrete categories, regression models output real-valued predictions.

While the penguins dataset is most commonly used for classification tasks, it can also be used for regression problems by selecting a continuous target variable. For example, we might be interested in predicting a penguin’s body mass based on its physical measurements like bill length, bill depth, flipper length, and other available features.

.. figure:: img/4-penguins-pairplot.png
   :align: center
   :width: 640px

Depending on model construction procedures, in this episode we explore a variety of regression algorithms to predict penguin body mass based on flipper length. These models are chosen to represent different categories of ML approaches, from simple to more complex and flexible methods.

- We begin with KNN regression, which makes predictions based on the average of the closest training samples. It’s a non-parametric, instance-based model that captures local patterns in the data.
- Next we apply linear models, such as standard Linear Regression and Regularized Regression, which assume a straight-line relationship between flipper length and body mass. These models are interpretable and efficient, making them a solid baseline for comparison.
- To address possible non-linear trends in the data, we incorporate non-linear models like Polynomial Regression with higher-degree terms and and Support Vector Regression (SVR) with RBF kernels
- Tree-based models, including decision trees, random forests, and gradient boosting, offer a robust alternative by recursively partitioning the feature space or building ensembles to improve accuracy and handle non-linearities effectively
- Finally, we explore neural networks as a universal function approximator, capable of learning intricate relationships but requiring larger datasets and computational resources.

Each model’s performance is rigorously assessed using cross-validated metrics (RMSE (root mean squared error), R²), and the corresponding predictive curve reveals how well they capture the biological allometry between flipper length and body mass.

This tiered approach -- from simple models like linear regression to more complex ones such as random forests and neural networks -- ensures that we balance interpretability with predictive power. By progressing through these levels of model complexity, we aim to identify the most suitable algorithm for accurately predicting penguin body mass from flipper length, while maintaining an understanding of how each model interprets the relationship between features and target.



Data Preparation
----------------

From the pairplot, we can see that the relationship between body mass and flipper length is visually strong, suggesting a clear positive correlation (figure below) between these two variables. Therefore, we use this pair for the regression task, as their strong relationship makes them suitable for modeling and predicting body mass based on flipper length. By modeling this relationship, we aim to estimate body mass based on flipper length, which can be valuable in ecological studies and predictive modeling involving penguin morphology.

.. figure:: img/5-penguins-bodyMass-flipperLength.png
   :align: center
   :width: 512px


Following the procedures adopted in the previous episode, we begin by importing the Penguins dataset and performing data preprocessing, including handling missing values and outliers. For the regression task, categorical features are not required, so there is no need to encode them.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import seaborn as sns

   penguins = sns.load_dataset('penguins')

   # remove missing values
   penguins_regression = penguins.dropna()

   # check duplicate values from dataset
   penguins_regression.duplicated().value_counts()

   # calculate lower and upper limit of outlier using IQR method
   IQR = penguins_regression["body_mass_g"].quantile(0.75) - penguins_regression["body_mass_g"].quantile(0.25)
   lower_limit = penguins_regression["body_mass_g"].quantile(0.25) - (1.5 * IQR)
   upper_limit = penguins_regression["body_mass_g"].quantile(0.75) + (1.5 * IQR)

   print(f"Body Mass:      lower limt of IQR = {lower_limit:.2f} and upper limit of IQR = {upper_limit:.2f}")

   IQR = penguins_regression["flipper_length_mm"].quantile(0.75) - penguins_regression["flipper_length_mm"].quantile(0.25)
   lower_limit = penguins_regression["flipper_length_mm"].quantile(0.25) - (1.5 * IQR)
   upper_limit = penguins_regression["flipper_length_mm"].quantile(0.75) + (1.5 * IQR)

   print(f"Flipper Length: lower limt of IQR = {lower_limit:7.2f} and upper limit of IQR = {upper_limit:7.2f}")

Next, we separate the dataset into features (flipper length) and labels (body mass), and then split it into training and testing sets. This is followed by feature scaling to standardize the data before model training.

.. code-block:: python

   X = penguins_regression[["flipper_length_mm"]].values
   y = penguins_regression["body_mass_g"].values

   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

   # standardize features
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)



Training Model & Evaluating Model Performance
---------------------------------------------


k-Nearest Neighbors (KNN)
^^^^^^^^^^^^^^^^^^^^^^^^^

We begin by applying the KNN algorithm to the penguin regression task, with a code example provided below.

.. code-block:: python

   from sklearn.neighbors import KNeighborsRegressor

   knn_model = KNeighborsRegressor(n_neighbors=5)
   knn_model.fit(X_train_scaled, y_train)

   # predict on test data
   y_pred_knn = knn_model.predict(X_test_scaled)

   # evaluate model performance
   from sklearn.metrics import root_mean_squared_error, r2_score
   rmse_knn = root_mean_squared_error(y_test, y_pred_knn)
   r2_value_knn = r2_score(y_test, y_pred_knn)
   print(f"K-Nearest Neighbors RMSE: {rmse_knn:.2f}, R²: {r2_value_knn:.2f}")


In order to visualize the KNN algorithm on the regression task, we plot the **predictive curve** that maps input values to predicted outputs. This curve shows how K-Nearest Neighbors responds to changes in a single feature. Since KNN is a non-parametric, instance-based method, it doesn't learn a fixed equation during training. Instead, predictions are based on averaging the target values of the k nearest training examples for any given input.

The resulting predictive curve is typically piecewise-smooth, adapting to local patterns in the data, that is, the curve may bend or flatten in response to regions where data is dense or sparse.

.. figure:: img/5-regression-predictive-curve-knn-5.png
   :align: center
   :width: 512px

This makes the predictive curve an especially useful tool for understanding whether KNN is underfitting (*e.g.*, when k is large) or overfitting (*e.g.*, when k is small). By adjusting k and observing the changes in the curve’s shape, we can intuitively tune the model’s bias-variance tradeoff.

.. figure:: img/5-regression-predictive-curve-knn-1357.png
   :align: center
   :width: 512px








