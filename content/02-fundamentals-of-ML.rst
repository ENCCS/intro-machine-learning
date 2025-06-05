Fundamentals of Machine Learning
================================


.. questions::

   - what are the main types of MLs?



.. objectives::

   - Describe main types of ML



Types of Machine Learning
-------------------------


ML can be broadly categorized into three main types depending on how the models learn from input data and the nature of the input data they process.


Supervised learning
^^^^^^^^^^^^^^^^^^^

In supervised learning, the model is trained on a labeled dataset, where each input is paired with a corresponding output (label). The goal is to learn a mapping from inputs to outputs to make predictions on new, unseen data.

Supervised learning has two subtypes: **Classification** (predicting discrete categories) and **Regression** (predicting continuous values).

Here are representative examples of these two subtypes in real-word problems:

- **Classification**: email spam detection (spam/ham), image recognition (cat/dog), medical diagnosis (disease/no disease).
- **Regression**: house price prediction, weather forecasting.


Unsupervised learning
^^^^^^^^^^^^^^^^^^^^^

In unsupervised learning, the model works with unlabeled data, identifying patterns, structures, or relationships within the data without explicit guidance on what to predict.

Unsupervised learning also has two subtypes: **Clustering** (grouping similar data points together) and **Dimensionality reduction** (simplifying data by reducing features while preserving important information)

Representative examples of these two subtypes in real-word problems:

- **Clustering**: customer segmentation in marketing (grouping users by behavior), image segmentation (grouping similar pixels).
- **Dimensionality reduction**: compressing high-dimensional data (*e.g.*, reducing image features for faster processing), anomaly detection.


Reinforcement learning
^^^^^^^^^^^^^^^^^^^^^^

The model (agent) learns by interacting with an environment. It takes actions, receives feedback (rewards or penalties), and learns a strategy (policy) to maximize long-term rewards.

Representative examples of reinforcement learning in real-word problems: game-playing AI (*e.g.*, AlphaGo), robot navigation, autonomous driving.


.. figure:: img/ML-three-types.png
   :align: center
   :width: 512px

   Three main types of machine learning. Main approaches include classification and regression under the supervised learning and clustering under the unsupervised learning. Reinforcement learning enhance the model performance by interacting with environment. Coloured dots and triangles represent the training data. Yellow stars represent the new data which can be predicted by the trained model. This figure was taken from the paper `Machine Learning Techniques for Personalised Medicine Approaches in Immune-Mediated Chronic Inflammatory Diseases: Applications and Challenges <https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2021.720694/full>`_.


Other subtypes
^^^^^^^^^^^^^^

In addition to supervised and unsupervised learning, there are other important paradigms in machine learning.

- **Semi-supervised learning** bridges the gap between supervised and unsupervised learning by using a small amount of labeled data together with a large amount of unlabeled data, helping models learn more effectively when labeling is expensive or time-consuming (*e.g.*, medical image analysis).
- **Self-supervised learning** is a form of unsupervised learning where the model generates its own labels from the data -- typically for pretraining models on tasks like image or language understanding, enabling them to learn robust representations without explicit labels (*e.g.*, predicting the next word in a sentence, and filling in missing image patches)
- **Transfer learning** involves applying knowledge from a pretrained model, trained on a large, general dataset, to a new, related task, significantly reducing training time and data requirements (*e.g.*, fine-tuning a speech recognition model for a new dialect).

These techniques expand the capabilities and versatility of machine learning across data-limited or computationally constrained environments.



Machine Learning Workflow
-------------------------


What is a workflow for ML?
^^^^^^^^^^^^^^^^^^^^^^^^^^

A machine learning workflow is a structured approach for developing, training, evaluating, and deploying machine learning models. It typically involves several key phases, including data collection, preprocessing, model training and evaluation, and finally, deployment to production.

Here is a graphical representation of ML workflow, and a concise overview of the key steps are described below.






Data collection and preprocessing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ML, data collection and preprocessing are crucial steps that significantly affect the performance of a model. High-quality, well-processed data leads to better predictions, while poor data can result in unreliable models.

- **Data collection**: Gather the necessary data from various sources (*e.g.*, databases, APIs (twitter, linkedin, *etc.*), or manual collection), and ensure that data is representative and sufficient for the problem.
- **Data preprocessing**: Clean and prepare data by handling missing values (drop, impute, or predict), removing duplicates or irrelevant data, fixing inconsistencies (*e.g.*, "USA" vs. "United States"), normalizing/scaling features, encoding categorical variables, and addressing outliers, and other data quality issues.
- **Exploratory data analysis (EDA)**: Analyze data to uncover distributions, correlations, patterns, anomalies, and insights using visualizations and statistical methods. This helps in feature selection and understanding data distribution.
- **Feature engineering**: Create or select relevant features to improve model performance. This may involve dimensionality reduction (*e.g.*, PCA (principal component analysis)) or creating new features based on domain knowledge.
- **Data splitting**: Divide the dataset into training, validation, and test sets (*e.g.*, 70-15-15 split) to evaluate model performance and prevent overfitting.



Model selection and training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model Selection and Training refer to the process of choosing an appropriate model architecture and training it to learn patterns from data to solve a specific task. It involves selecting the appropriate algorithms (*e.g.*, linear/logistic regression, decision trees, neural networks, Gradient Boosting) based on the problem type, configuring its hyperparameters, and optimizing its parameters using training data to minimize error or maximize performance metrics.


