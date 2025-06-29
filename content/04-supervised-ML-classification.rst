Supervised Learning (I): Classification
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

.. figure:: img/penguins-categories.png
   :align: center
   :width: 512px

   The Palmer Penguins data were collected from 2007-2009 by Dr. Kristen Gorman with the `Palmer Station Long Term Ecological Research Program <https://lternet.edu/site/palmer-antarctica-lter/>`_, part of the `US Long Term Ecological Research Network <https://lternet.edu/>`_. The data were imported directly from the `Environmental Data Initiative (EDI) <https://edirepository.org/>`_ Data Portal, and are available for use by CC0 license (“No Rights Reserved”) in accordance with the `Palmer Station Data Policy <https://lternet.edu/data-access-policy/>`_.



Importing Dataset
-----------------

Seaborn provides the Penguins dataset through its built-in data-loading functions. We can access it using ``sns.load_dataset('penguin')`` and then have a quick look at the data:

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import pandas as pd
   import seaborn as sns

   penguins = sns.load_dataset('penguins')
   penguins


.. csv-table::
   :widths: auto
   :delim: ;

   ; species; island; bill_length_mm; bill_depth_mm; flipper_length_mm; body_mass_g; sex
   0; Adelie; Torgersen; 39.1; 18.7; 181.0; 3750.0; Male
   1; Adelie; Torgersen; 39.5; 17.4; 186.0; 3800.0; Female
   2; Adelie; Torgersen; 40.3; 18.0; 195.0; 3250.0; Female
   3; Adelie; Torgersen; NaN; NaN; NaN; NaN; NaN
   4; Adelie; Torgersen; 36.7; 19.3; 193.0; 3450.0; Female
   ...; ...; ...; ...; ...; ...; ...; ...
   339; Gentoo; Biscoe; NaN; NaN; NaN; NaN; NaN
   340; Gentoo; Biscoe; 46.8; 14.3; 215.0; 4850.0; Female
   341; Gentoo; Biscoe; 50.4; 15.7; 222.0; 5750.0; Male
   342; Gentoo; Biscoe; 45.2; 14.8; 212.0; 5200.0; Female
   343; Gentoo; Biscoe; 49.9; 16.1; 213.0; 5400.0; Male


There are seven columns include:

- *species*: penguin species (Adelie, Chinstrap, Gentoo)
- *island*: island where the penguin was found (Biscoe, Dream, Torgersen)
- *bill_length_mm*: length of the bill
- *bill_depth_mm*: depth of the bill
- *flipper_length_mm*: length of the flipper
- *body_mass_g*: body mass in grams
- *sex*: male or female

Looking at numbers from ``penguins`` and ``penguins.describe()`` usually does not give a very good intuition about the data we are working with, we have the preference to visualize the data.

One nice visualization for datasets with relatively few attributes is the Pair Plot, which can be created using ``sns.pairplot(...)``.
It shows a scatterplot of each attribute plotted against each of the other attributes.
By using the ``hue='species'`` setting for the pairplot the graphs on the diagonal are layered kernel density estimate plots for the different values of the ``species`` column.

.. code-block:: python

   sns.pairplot(penguins_classification[["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]], hue="species", height=2.0)

.. figure:: img/penguins-pairplot.png
   :align: center
   :width: 640px


.. challenge:: Discussion

   Take a look at the pairplot we created. Consider the following questions:

   - Is there any class that is easily distinguishable from the others?
   - Which combination of attributes shows the best separation for all 3 class labels at once?
   - (optional) Create a similar pairplot, but with ``hue="sex"``. Explain the patterns you see. Which combination of features distinguishes the two sexes best?

   .. solution::

     1. The plots show that the green class (Gentoo) is somewhat more easily distinguishable from the other two.
     2. Adelie and Chinstrap seem to be separable by a combination of bill length and bill depth (other combinations are also possible such as bill length and flipper length).
     3. ``sns.pairplot(penguins_classification, hue="sex", height=2.0)``. From the plots you can see that for each species females have smaller bills and flippers, as well as a smaller body mass. You would need a combination of the species and the numerical features to successfully distinguish males from females. The combination of bill_depth_mm and body_mass_g gives the best separation.



Data Processing
---------------


Handling missing values and outliers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a ML task, the input data (features) and target data (label) are not yet in a right format to use. We need to pre-process the data (as what we did yesterday) to clean missing values using ``penguins_classification = penguins.dropna()`` and check duplicate values using ``penguins_classification.duplicated().value_counts()``.

It is noted that we don't have outliers in this dataset (as we have discussed this issue in the `data processing <>`_ tutorial). For the other datasets you use for the first time, you should check if there are outliers for some features in the dataset, and then take steps to handle the outliers, either to imputate outliers with mean/median values or to remove abnormal outliers for simplicity.


Encoding categorical variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the classification task, we will use the categorical variable *species* as the label (target variable), and other columns as features to predict the species of penguins.

.. challenge:: Discussion

   - why to use *species*?
   - why not to use other other categorical variables (here it would be *island* and *sex*)?

   .. solution::

     1. *species* will be the main biological classification target in this dataset as it 3 distinct classes (Adelie, Chinstrap, and Gentoo). This is commonly used in ML tutorials as a multi-class classification example (similar to the `Iris dataset <https://archive.ics.uci.edu/dataset/53/iris>`_).
     2. *island* is not a ideal label as it is just geographical info, not a biological classification target; *sex* is possible but quite limited. This variable only has two classes (only for binary classification), and the data is unbalanced and has missing values.


It is noted that ML models cannot directly process categorical (non-numeric) data, so we have to encode categorical variables like *species*, *island*, and *sex* into numerical values. Here we use ``LabelEncoder`` from ``sklearn.preprocessing`` to convert the species column, which serves as our classification target. The ``LabelEncoder`` assigns a unique integer to each species: "Adelie" becomes 0, "Chinstrap" becomes 1, and "Gentoo" becomes 2. This transformation allows classification algorithms to treat the species labels as distinct, unordered classes.

Then we apply the same rule to encode the island and sex columns. Although these are typically better handled with one-hot encoding due to their nominal nature, we use ``LabelEncoder`` here for simplicity and compact representation. Each unique category in island (*e.g.*, "Biscoe", "Dream", "Torgersen") and sex (*e.g.*, "Male", "Female") is mapped to a unique integer. This enables us to include them as input features in the model without manual transformation. However, it’s important to note that ``LabelEncoder`` introduces an implicit ordinal relationship, which might not always be appropriate -- in such cases, ``OneHotEncoder`` is preferred.

.. code-block:: python

   from sklearn.preprocessing import LabelEncoder

   encoder = LabelEncoder()

   # encode "species" column with 0=Adelie, 1=Chinstrap, and 2=Gentoo
   penguins_classification.loc[:, 'species'] = encoder.fit_transform(penguins_classification['species'])

   # encode "island" column with 0=Biscoe, 1=Dream and 2=Torgersen
   penguins_classification.loc[:, 'island'] = encoder.fit_transform(penguins_classification['island'])

   # encode "sex" column 0=Female and 1=Male
   penguins_classification.loc[:, 'sex'] = encoder.fit_transform(penguins_classification['sex'])


Data Splitting
--------------


Splitting features and labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In preparing the penguins dataset for classification, we first need to split the data into features and labels. The target variable we aim to predict is the penguin species, which we encode into numeric labels using ``LabelEncoder``. This encoded species column will be the **label vector** (*e.g.*, **y**). The remaining columns -- such as bill length, bill depth, flipper length, body mass, and encoded categorical variables like island and sex -- constitute the **feature matrix** (*e.g.*, **X**). These features contain the input information the model will learn from.

Separating features (X) from labels (y) ensures a clear distinction between what the model uses for prediction and what it is trying to predict.


.. code-block:: python

   X = penguins_classification.drop(['species'], axis=1)
   y = penguins_classification['species'].astype('int')


Splitting training and testing sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

After separating features and labels in the penguins dataset, we further divide the data into a training set and a testing set. The training set is used to train the model, allowing it to learn patterns and relationships from the data, and the test set, on the other hand, is reserved for evaluating the model’s performance on unseen data. A common split is 80% for training and 20% for testing, which provides enough data for training while still retaining a meaningful test set.

This splitting is typically done using the ``train_test_split`` function from ``sklearn.model_selection``, with a fixed ``random_state`` to ensure reproducibility.

.. code-block:: python

   from sklearn.model_selection import train_test_split

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

   print(f"Number of examples for training is {len(X_train)} and test is {len(X_test)}")


Feature scaling
^^^^^^^^^^^^^^^

Before training, it is also essential to ensure that numerical features are properly scaled via applying standardization or normalization -- especially for distance-based or gradient-based models -- to achieve optimal results.

.. code-block:: python

   from sklearn.preprocessing import StandardScaler
   
   # Standardize features
   scaler = StandardScaler()

   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)



Training Model & Evaluating Model Performance
---------------------------------------------


After preparing the Penguins dataset by handling missing values, encoding categorical variables, and splitting it into features-labels and training-test datasets, the next step is to apply classification algorithms including k-Nearest Neighbors (KNN), Decision Trees, Random Forests, Naive Bayes, and Neural Networks to predict penguin species based on their physical measurements
Each algorithm offers a unique approach to pattern recognition and generalization, and applying them to the same prepared dataset allows for a fair comparison of their predictive performance.

Below is the generic steps for representative algorithms we will use to training a model for penguins classification:

- choosing a model class and importing that model ``from sklearn.neighbors import XXXClassifier``
- choosing the model hyperparameters by instantiating this class with desired values ``xxx_clf = XXXClassifier(<... hyperparameters ...>)``
- training the model to the preprocessed train data by calling the ``fit()`` method of the model instance ``xxx_clf.fit(X_train_scaled, y_train)``
- making predictions using the trained model on test data ``y_pred_xxx = xxx_clf.predict(X_test_scaled)``
- evaluating model’s performance using available metrics ``score_xxx = accuracy_score(y_test, y_pred_xxx)``
- (optional) data visualization of confusion matrix and relevant data


k-Nearest Neighbors (KNN)
^^^^^^^^^^^^^^^^^^^^^^^^^

One intuitive and widely-used method is the k-Nearest Neighbors (KNN) algorithm. KNN is a non-parametric, instance-based algorithm that predicts a sample's label based on the majority class of its *k* closest neighbors in training set.

KNN does not require training in the traditional sense; instead, it stores the entire dataset and performs computation during prediction time. This makes it a lazy learner but potentially expensive during inference.

Here is an example of using the KNN algorithm to determine which class the new point belongs to. When the given query point, the KNN algorithm calculates the distance between this point and all points in the training dataset. It then selects the *k* points that are closest. The class with the most representatives among the *k* neighbors is chosen to be the prediction result for the query point.
It is noted that the choice of *k* (the number of neighbors) significantly affects performance: a small *k* may be sensitive to noise, while a large *k* may smooth over important patterns.

.. figure:: img/knn-example.png
   :align: center
   :width: 640px


Let’s create the KNN model. Here we choose 3 as the *k* value of the algorithm, which means that data needs 3 neighbors to be classified as one entity. Then we fit the train data using the ``fit()`` method.

.. code-block:: python

   from sklearn.neighbors import KNeighborsClassifier

   knn_clf = KNeighborsClassifier(n_neighbors=3)
   knn_clf.fit(X_train_scaled, y_train)


After we fitting the training data, we use the trained model to predict species on the test set and evaluate its performance.

For classification tasks, metrics like accuracy, precision, recall, and the F1-score provide a comprehensive view of model performance.

- **accuracy** measures the proportion of correctly classified instances across all species (Adelie, Chinstrap, Gentoo), and it gives an overall measure of how often the model is correct, but it can be misleading for imbalanced datasets.
- **precision** quantifies the proportion of correct positive predictions for each species, while **recall** assesses the proportion of actual positives correctly identified.
- the **F1-score**, the harmonic mean of precision and recall, balances these metrics for each class, especially useful given the dataset’s imbalanced species distribution.


.. code-block:: python

   # predict on test data
   y_pred_knn = knn_clf.predict(X_test_scaled)

   # evaluate model performance
   from sklearn.metrics import classification_report, accuracy_score

   score_knn = accuracy_score(y_test, y_pred_knn)

   print("Accuracy for k-Nearest Neighbors:", score_knn)
   print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))


In classification tasks, a **confusion matrix** is a valuable tool for evaluating model performance by comparing predicted labels against true labels.
For a multiclass classification task like the penguins dataset, the confusion matrix is an **N x N** matrix, where **N** is the number of target classes (here **N=3** for three penguins species). Each cell *(i, j)* in the matrix indicates the number of instances where the true class was *i* and the model predicted class *j*. Diagonal elements represent correct predictions, while off-diagonal elements indicate misclassifications. The confusion matrix provides an easy-to-understand overview of how often the predictions match the actual labels and where the model tends to make mistakes.

Since we will plot the confusion matrix multiple times, we write a function and call this function later whenever needed, which promotes clarity and avoids redundancy. This is especially helpful as we evaluate multiple classifiers such as KNN, Decision Trees, or SVM on the penguins dataset.

.. code-block:: python

   from sklearn.metrics import confusion_matrix

   def plot_confusion_matrix(conf_matrix, title, fig_name):
       plt.figure(figsize=(6, 5))
       sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd',
                   xticklabels=["Adelie", "Chinstrap", "Gentoo"],
                   yticklabels=['Adelie', 'Chinstrap', 'Gentoo'], cbar=True)
       
       plt.xlabel("Predicted Label")
       plt.ylabel("True Label")
       plt.title(title)
       plt.tight_layout()
       plt.savefig(fig_name)

We compute the confusion matrix from the trined model using the KNN algorithm, and visualize the matrix.

.. code-block:: python

   cm_knn = confusion_matrix(y_test, y_pred_knn)

   plot_confusion_matrix(cm_knn, "Confusion Matrix using KNN algorithm", "confusion-matrix-knn.png")


.. figure:: img/confusion-matrix-knn.png
   :align: center
   :width: 420px

   The first row: there are 28 Adelie penguins in the test data, and all these penguins are identified as Adelie (valid). The second row: there are 20 Chinstrap pengunis in the test data, with 2 identified as Adelie (invalid), none are correctly recognized as Chinstrap, and 18 identified as Chinstrap (valid). The third row: there are 19 Gentoo penguins in the test data, and all these penguins are identified as Gentoo (valid).



Logistic Regression
^^^^^^^^^^^^^^^^^^^

**Logistic Regression** is a fundamental classification algorithm to predict categorical outcomes.
Despite its name, logistic regression is not a regression algorithm but a classification method that predicts the probability of an instance belonging to a particular class.

For binary classification, it uses the logistic (**sigmoid**) function to map a linear combination of input features to a probability between 0 and 1, which is then thresholded (typically at 0.5) to assign a class.

For a multiclass classification, logistic regression can be extended using strategies like **one-vs-rest** (OvR) or softmax regression.

- in OvR, a separate binary classifier is trained for each species against all others.
- **softmax regression** generalizes the logistic function to compute probabilities across all classes simultaneously, selecting the class with the highest probability.

.. figure:: img/logistic-regression-example.png
   :align: center
   :width: 640px

   (Upper left) the sigmoid function; (upper middle) the softmax regression process: three input features to the softmax regression model resulting in three output vectors where each contains the predicted probabilities for three possible classes; (upper right) a bar chart of softmax outputs in which each group of bars represents the predicted probability distribution over three classes; lower subplots) three binary classifiers distinguish one class from the other two classes using the one-vs-rest approach.


The creation of a Logistic Regression model and the process of fitting it to the training data are nearly identical to those used for the KNN model described above, except that a different classifier is selected. The code example and the resulting confusion matrix plot are provided below:

.. code-block:: python

   from sklearn.linear_model import LogisticRegression

   lr_clf = LogisticRegression(random_state = 0)
   lr_clf.fit(X_train_scaled, y_train)

   y_pred_lr = lr_clf.predict(X_test_scaled)

   score_lr = accuracy_score(y_test, y_pred_lr)
   print("Accuracy for Logistic Regression:", score_lr )
   print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

   cm_lr = confusion_matrix(y_test, y_pred_lr)
   plot_confusion_matrix(cm_lr, "Confusion Matrix using Logistic Regression algorithm", "confusion-matrix-lr.png")

.. figure:: img/confusion-matrix-lr.png
   :align: center
   :width: 420px



Naive Bayes 
^^^^^^^^^^^

The **Naive Bayes** algorithm is a simple yet powerful probabilistic classifier based on Bayes' Theorem. This classifier assumes that all features are equally important and independent which is often not the case and may result in some bias. However, the assumption of independence simplifies the computations by turning conditional probabilities into products of probabilities. This algorithm computes the probability of each class given the input features and selects the class with the highest posterior probability. 

Logistic regression and Naive Bayes are both popular algorithms for classification tasks, but they differ significantly in their approach, assumptions, and mechanics.

- Logistic regression is a **discriminative** model that directly models the probability of a data point belonging to a particular class by fitting a linear combination of features through a logistic (sigmoid) function for binary classification or softmax for multiclass tasks. For the penguins dataset, it would use features like bill length and flipper length to compute a weighted sum, transforming it into probabilities for species like Adelie, Chinstrap, or Gentoo. It assumes a linear relationship between features and the log-odds of the classes and optimizes parameters using maximum likelihood estimation, making it sensitive to feature scaling and correlations. Logistic regression is robust to noise and can handle correlated features to some extent, but it may struggle with highly non-linear relationships unless feature engineering is applied.
- Naive Bayes, in contrast, is a **generative** model that relies on Bayes’ theorem to compute the probability of a class given the features, assuming conditional independence between features given the class. For the penguins dataset, it would estimate the likelihood of features (*e.g.*, bill depth) for each species and combine these with prior probabilities to predict the most likely species. The "naive" assumption of feature independence often doesn’t hold (*e.g.*, bill length and depth may be correlated), but Naive Bayes is computationally efficient, works well with high-dimensional data, and is less sensitive to irrelevant features. However, it can underperform when feature dependencies are significant or when the data distribution deviates from its assumptions (*e.g.*, Gaussian for continuous features in Gaussian Naive Bayes). Unlike logistic regression, it doesn’t require feature scaling but may need careful handling of zero probabilities (*e.g.*, via smoothing).

Below is an example comparing Logistic Regression and Naive Bayes decision boundaries on a synthetic dataset having two features. The visualization highlights their fundamental differences in modeling assumptions and classification behavior: **Logistic Regression learns a linear decision boundary directly, while Naive Bayes models feature distributions per class (assuming independence)**.

.. figure:: img/4-naive-bayes-example.png
   :align: center
   :width: 640px

To apply Naive Bayes, we use ``GaussianNB`` from ``sklearn.naive_bayes``, which assumes that the features follow a Gaussian (normal) distribution, which is an appropriate choice for continuous numerical data such as bill length and body mass. Since Naive Bayes relies on probabilities, **feature scaling is not required**, but **handling missing values and encoding categorical variables numerically is still necessary**.

While Naive Bayes may not outperform more complex models like Random Forests, it offers **fast training, low memory usage**, and good performance for simple tasks.

.. code-block:: python

   from sklearn.naive_bayes import GaussianNB

   nb_clf = GaussianNB()
   nb_clf.fit(X_train_scaled, y_train)

   y_pred_nb = nb_clf.predict(X_test_scaled)

   score_nb = accuracy_score(y_test, y_pred_nb)
   print("Accuracy for Naive Bayes:", score_nb)
   print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

   cm_nb = confusion_matrix(y_test, y_pred_nb)
   plot_confusion_matrix(cm_nb, "Confusion Matrix using Naive Bayes algorithm", "confusion-matrix-nb.png")

.. figure:: img/4-confusion-matrix-nb.png
   :align: center
   :width: 420px



Support Vector Machine (SVM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Previously we shown an example using Logistic Regression classifier producing a linear decision boundary that separates cats from dogs. It works by fitting a linear decision boundary that separates two classes based on the logistic function, making it particularly effective when the data is linearly separable. One characteristic of logistic regression is that the decision boundary tends to fall in the region where the probabilities of two classes are closest -- typically where the model is most uncertain.

However, when there exists a large gap between two well-separated classes -- as often occurs when distinguishing cats and dogs based on weight and ear length -- logistic regression faces an inherent limitation: infinite possible solutions. The algorithm has no mechanism to select an "optimal" boundary when multiple valid linear separators exist in the wide margin between classes, and it will place the decision boundary somewhere in that gap, leading to a broad, undefined decision region with no supporting data. While this may not affect accuracy on clearly separated data, it can make the model less robust when new or noisy data appears near that boundary.

Below is an example, again, to separate cats from dogs based on ear length and weight. Besides the linear decision boundary from Logistic Regression classifier, we can find three additional linear boundaries that can also have a good separation of cats from dogs. Which one is better than the others and how to evaluate their performance on unseen data?

.. figure:: img/4-svm-example-large-gap.png
   :align: center
   :width: 640px

To better handle such situation, we can transition to the **Support Vector Machine** (SVM) algorithm. SVM takes a different approach by focusing on the concept of maximizing the margin -- the distance between the decision boundary and the closest data points from each class (the support vectors) (as shown in the figure below). When there is a large gap between the two classes, SVM utilizes that space effectively by pushing the boundary toward the center of the gap while maintaining the maximum margin. This leads to a more stable and robust classifier, particularly in cases where the classes are well-separated.

Unlike Logistic Regression, which uses all data points to estimate probabilities, SVM relies primarily on the most critical examples (the ones nearest the boundary), making it less sensitive to outliers and more precise in defining class divisions.

.. figure:: img/4-svm-example-with-max-margin-separation.png
   :align: center
   :width: 640px

   The SVM classification boundary for distinguishing cats and dogs based on ear length and weight. The solid black line represents the maximum margin hyperplane (decision boundary), while the dashed green lines show the positive and negative hyperplanes that define the margin. Black circles highlight the support vectors - the critical data points that determine the margin width.

To apply SVM, we use ``SVC`` (Support Vector Classification) from ``sklearn.svm``, which by default assumes that the features follow a nonlinear relationship modeled by the ``rbf`` (Radial Basis Function) kernel. This kernel allows the model to find complex decision boundaries by implicitly mapping the input features into a higher-dimensional space. You can easily change the kernel to ``linear``, ``poly``, or ``sigmoid`` to experiment with different decision boundaries.

By adjusting the hyperparameters such as ``C`` (regularization strength) and ``gamma`` (kernel coefficient), we can control the trade-off between the margin width and classification accuracy. Below is a code example demonstrating how to use SVC with the RBF kernel for the penguins classification task.

.. code-block:: python

   from sklearn.svm import SVC

   svm_clf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=0)
   svm_clf.fit(X_train_scaled, y_train)

   y_pred_svm = svm_clf.predict(X_test_scaled)

   score_svm = accuracy_score(y_test, y_pred_svm)
   print("Accuracy for Support Vector Machine:", score_svm)
   print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

   cm_svm = confusion_matrix(y_test, y_pred_svm)
   plot_confusion_matrix(cm_svm, "Confusion Matrix using Support Vector Machine algorithm", "confusion-matrix-svm.png")



Decision Tree
^^^^^^^^^^^^^

**Decision Tree** algorithm is a versatile and interpretable method for classification tasks. The core idea of this algorithm is to recursively split the dataset into smaller subsets based on feature thresholds creating a tree-like structure of decisions that result in the most significant separation of target classes.


Here is one example showing how to separate cats and dogs on the basis of two or three features.

.. figure:: img/4-decision-tree-example.png
   :align: center
   :width: 640px

   (Upper) decision boundary separating cats and dogs based on two features (ear length and weight), and the corresponding decision tree structure; (lower): two decision boundaries separating cats and dogs based on three features (ear length, weight, and tail length), and the corresponding decision tree structure.

The code example for the Decision Tree classifier is provided below.

.. code-block:: python

   from sklearn.tree import DecisionTreeClassifier

   dt_clf = DecisionTreeClassifier(max_depth=3, random_state = 0)
   dt_clf.fit(X_train_scaled, y_train)

   y_pred_dt = dt_clf.predict(X_test_scaled)

   score_dt = accuracy_score(y_test, y_pred_dt)
   print("Accuracy for Decision Tree:", score_dt )
   print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

   cm_dt = confusion_matrix(y_test, y_pred_dt)
   plot_confusion_matrix(cm_dt, "Confusion Matrix using Decision Tree algorithm", "confusion-matrix-dt.png")


We visualize the Decision Tree structure to understand how penguins are classified based on their physical characteristics.

.. code-block:: python

   from sklearn.tree import plot_tree

   plt.figure(figsize=(16, 6))
   plot_tree(dt_clf, feature_names=X.columns, filled=True, rounded=True, fontsize=10)

   plt.title("Decision Tree Structure for Penguins Species Classification", fontsize=18)

.. figure:: img/4-decision-tree-structure.png
   :align: center
   :width: 640px
