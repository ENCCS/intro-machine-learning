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

Looking at numbers from `penguins` `penguins.describe()` usually does not give a very good intuition about the data we are working with, we have the preference to visualize the data.

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

   # encode `species` column with 0=Adelie, 1=Chinstrap, and 2=Gentoo
   penguins_classification.loc[:, 'species'] = encoder.fit_transform(penguins_classification['species'])

   # encode `island` column with 0=Biscoe, 1=Dream and 2=Torgersen
   penguins_classification.loc[:, 'island'] = encoder.fit_transform(penguins_classification['island'])

   # encode `sex` column 0=Female and 1=Male
   penguins_classification.loc[:, 'sex'] = encoder.fit_transform(penguins_classification['sex'])


Data Splitting
--------------


Splitting features and labels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In preparing the penguins dataset for classification, we first need to split the data into features and labels. The target variable we aim to predict is the penguin species, which we encode into numeric labels using ``LabelEncoder``. This encoded species column will be the **label vector** (*e.g.*, **y**). The remaining columns -- such as bill length, bill depth, flipper length, body mass, and encoded categorical variables like island and sex -- constitute the **feature matrix** (*e.g.*, **X**). These features contain the input information the model will learn from.

Separating features (X) from labels (y) ensures a clear distinction between what the model uses for prediction and what it is trying to predict.


.. code-block:: python

   X = penguins_classification.drop(['species'], axis=1)
   y = penguins_classification['species']




