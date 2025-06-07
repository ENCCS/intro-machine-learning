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


Problem definition and project setup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Problem Definition** is the first and most critical phase of any ML project. It sets the direction, scope, and goals for the entire project.

- we should understand the problem domain: what is the real-world problem we are trying to solve? are we predicting, classifying, or grouping data? (*e.g.*, predict house prices, detect spam emails, cluster customers)
- we should determine if ML is the appropriate solution for the problem
- we then should identify the expected outputs: what will the ML model produce? (*e.g.*, a number, a label, or a probability)
- we define the type of ML task (*e.g.*, classification and regression tasks for supervised learning, clustering, dimensionality reduction for unsupervised learning, and decision-making tasks for reinforcement learning)


**Project Setup** is to set up the programming/development environment for the project.

- hardware requirements (CPU, SSD, GPU, cloud platforms, *etc.*)
- software requirements (programming languages and libraries, ML/DL frameworks, and development tools, IDEs, Git/Docker, *etc.)
- project structure: organize your project for clarity and scalability

A typical ML project structure looks like this

.. code-block:: console

  ML_Project/
  ├── data/                 # raw and processed data
  │   ├── raw/              # original, unprocessed data
  │   ├── processed/        # cleaned, preprocessed data
  ├── notebooks/            # jupyter notebooks for EDA & modeling
  ├── src/                  # source code
  │   ├── utils/            # utility functions (*e.g.*, metrics, logging)
  │   ├── preprocessing.py  # data cleaning script  
  │   └── train.py          # model training script
  ├── models/               # trained model files (*e.g.*, .pkl, .h5)
  ├── tests/                # unit and integration tests
  ├── README.md             # project overview and setup instructions
  ├── requirements.txt      # project dependencies
  ├── config.yaml           # configuration file for hyperparameters and paths


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


Model evaluation and assessment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Model evaluation and assessment in machine learning refers to the process of measuring and analyzing a model's performance to determine its effectiveness in solving a specific task. It involves using metrics and techniques to quantify how well the model generalizes to unseen data, identifies patterns, and meets desired objectives, typically using a test dataset separate from the training data.

Below are common evaluation metrics by task types:

.. list-table::  
   :widths: 100 100
   :header-rows: 1

   * - Task types 
     - Evaluation metrics
   * - Classification
     - Accuracy, precision, recall, F1-score, ROC-AUC, *etc.*
   * - Regression
     - Mean Squared Error (MSE), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R-squared, *etc.*
   * - Clustering
     - Silhouette score, Davies-Bouldin index, Calinski-Harabasz index.
   * - Ranking
     - Mean Reciprocal Rank (MRR), Normalized Discounted Cumulative Gain (NDCG).
   * - NLP or generative tasks
     - BLEU, ROUGE, perplexity (often overlaps with deep learning).

Here are representative techniques and processes for the assessment:

- **Train-validation-test split**: Divide data into training (model learning), validation (hyperparameter tuning), and test (final evaluation) sets to prevent overfitting.
- **Cross-validation**: Use k-fold cross-validation to assess model stability across multiple data subsets.
- **Confusion matrix**: For classification, visualize true positives, false negatives, etc.
- **Learning curves**: Plot training *vs.* validation performance to diagnose underfitting or overfitting.
- **Comparison with baselines**: Comparing model performance against simple baselines (*e.g.*, random guessing, linear models) to ensure meaningful improvement.
- **Robustness testing**: Evaluate performance under noisy, adversarial, or out-of-distribution data.
- **Fairness and bias analysis**: Assess model predictions for fairness across groups (*e.g.*, demographics).


Hyperparameter Tuning
^^^^^^^^^^^^^^^^^^^^^

Hyperparameter tuning is the process of optimizing the settings (hyperparameters) of a model that are not learned during training but significantly affect its performance. These include parameters like learning rate, number of hidden layers, or batch size, which control the model's behavior and training process.

The goal of this process is to find the best combination of hyperparameters that maximizes performance metrics (*e.g.*, accuracy, precision) on a validation set. 








