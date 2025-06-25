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









