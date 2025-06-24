Scientific Data for Machine Learning
====================================


.. questions::

   - why data is super important in ML?





Data is the Foundation of Machine Learning
------------------------------------------

Data is the backbone of ML as it serves as the foundation for training models to recognize patterns, make predictions, and generate insights. Without high-quality, relevant, and well-structured data, ML algorithms cannot learn meaningful patterns or make accurate predictions -- poor or insufficient data leads to biased, unreliable, or ineffective AI systems. From image recognition to natural language processing, every ML application depends on properly curated datasets for training, validation, and testing. In addition, data determines the applicability and scalability of ML solutions across domains, from scientific research to real-world applications.

Moreover, data preparation and processing consume a significant portion of the ML workflow, often more time than model development itself. Cleaning, transforming, and structuring raw data into a usable format ensures that algorithms can extract valuable insights efficiently. The choice of data formats, like CSV for simplicity or HDF5 for large-scale datasets, also impacts data storage, accessibility, and computational efficiency during model training and deployment.

In this episode , we will



Understanding Scientific Data
-----------------------------

Scientific data refers to any form of data that is collected, observed, measured, or generated as part of scientific research or experimentation. This data is used to support scientific analysis, develop theories, and validate hypotheses. It can come from a wide range of sources, including experiments, simulations, observations, or surveys across various scientific fields.

In general, scientific data can be described ty two terms: types of data and forms of data. They are related but distinct -- types describe the nature of the data, while forms describe the how the data is structured and formatted (and stored, which will be discussed below).


Types of scientific data
^^^^^^^^^^^^^^^^^^^^^^^^

Types of scientific data refer to what the data represents. It focuses on the nature or category of the data content.

- **Bit and byte**: The smallest unit of storage in a computer is a **bit**, which holds either a 0 or a 1. Typically, eight bits are grouped together to form a **byte**. A single byte (8 bits) can represent up to 256 distinct values. By organizing bytes in various ways, computers can interpret and store different types of data.
- **Numerical data**: Different numerical data types (*e.g.*, integer and floating-point numbers) require different binary representation. Using more bytes for each value increases the range or precision, but it consumes more memory.

	- For example, integers stored with 1 byte (8 bits) have a range from [-128, 127], while with 2 bytes (16 bits) the range becomes [-32768, 32767]. Integers are whole numbers and can be represented exactly given enough bytes.
	- In contrast, floating-point numbers (used for decimals) often suffer from representation errors, since most fractional values cannot be precisely expressed in binary. These errors can accumulate during arithmetic operations. Therefore, in scientific computing, numerical algorithms must be carefully designed to minimize error accumulation. To ensure stability, floating-point numbers are typically allocated 8 bytes (64 bits), keeping approximation errors small enough to avoid unreliable results.
	- In ML/DL, half, single, and double precision refer to different formats for representing floating-point numbers, typically using 16, 32, and 64 bits, respectively.
	
		- **Single precision** (32-bit) is commonly used as a balance between computational efficiency and numerical accuracy.
		- **Half precision** (16-bit) offers faster computation and reduced memory usage, making it popular for training large models on GPUs, though it may suffer from lower numerical stability.
		- **Double precision** (64-bit) provides higher accuracy but is slower and more memory-intensive, so it's mainly used when high numerical precision is critical.
		- Many modern frameworks, like TensorFlow and PyTorch, support mixed precision training, combining half and single precision to optimize performance while maintaining stability.

- **Text data**: When it comes to text data, the simplest character encoding is ASCII (American Standard Code for Information Interchange), which was the most widely used encoding until 2008 when UTF-8 took over. The original ASCII uses only 7 bits for representing each character and therefore can encode 128 specified characters. Later, it became common to use an 8-bit byte to store each character, resulting in extended ASCII with support for up to 256 characters. As computers became more powerful and the need for including more characters from other alphabets, UTF-8 became the most common encoding. UTF-8 uses a minimum of one byte and up to four bytes per character. This flexibility makes UTF-8 ideal for modern applications requiring global character support.
- **Metadata**: Metadata encompasses diverse information about data, including units, timestamps, identifiers, and other descriptive attributes. While most scientific data is either numerical or textual, the associated metadata is usually domain-specific, and different types of data may have different metadata conventions. In scientific applications, such as simulations and experimental results, metadata is typically integrated with the corresponding dataset to ensure proper interpretation and reproducibility.


Forms of scientific data
^^^^^^^^^^^^^^^^^^^^^^^^

Forms of scientific data refer to how the data is structured or formatted. It focuses on the presentation or shape of the data.

- **Tabular data structure** (numerical arrays) is a collection of numbers arranged in a specific structure that one can perform mathematical operations on. Examples of numerical arrays are scalar (0D), row or column vector (1D), matrix (2D), and tensor (3D), *etc.*
- **Textual data structure** is a format for storing and organizing text-based data. It represents unstructured or semi-structured information as sequences of characters (letters, numbers, symbols, punctuation) arranged in strings.
- **Images, videos, and audio** are forms of scientific data that represent information through visual and auditory formats. Images capture static visual information as pixel arrays, videos combine sequential frames to show temporal changes, and audio encodes sound signals as time-series data for analysis.
- **Graphs and networks** are forms of scientific data that represent relationships between entities as nodes and connections as edges. They are used to model complex systems such as social networks, molecular interactions, and ecological food webs, capturing the structure and connectivity of scientific phenomena.



Data Storage Format
-------------------


Representative data storage format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When it comes to data storage, there are many types of storage formats used in scientific computing and data analysis. There isn’t one data storage format that works in all cases, so choose a file format that best suits your data.

For tabular data, each column usually has a name and a specific data type while each row is a distinct sample which provides data according to each column (including missing values). The simplest way to save tabular data is using the so-called CSV (comma-separated values) file, which is human-readable and easily shareable. However, it is not the best format to use when working with big (numerical) data.

Gridded data is another very common data type in which numerical data is normally saved in a multi-dimensional grid (array). Common field-agnostic array formats include:

- **Hierarchical Data Format** (HDF5) is a high performance storage format for storing large amounts of data in multiple datasets in a single file. It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.
- **Network Common Data Form version 4** (NetCDF4) is a data format built on top of HDF5, but exposes a simpler API with a more standardised structure. NetCDF4 is one of the most used formats for storing large data from big simulations in physical sciences.
- **Zarr** is a data storage format designed for efficiently storing large, multi-dimensional arrays in a way that supports scalability, chunking, compression, and cloud-readiness.
- There are more file formats like `feather <https://arrow.apache.org/docs/python/feather.html>`_, `parquet <https://arrow.apache.org/docs/python/parquet.html>`_, `xarray <https://docs.xarray.dev/en/stable/>`_ and `npy <https://numpy.org/doc/stable/reference/routines.io.html>`_ to store arrow tables or data frames.



Overview of data storage format
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is an overview of common data formats (✅ for *good*, 🟨 for *ok/depends on a case*, and ❌ for *bad*) adapted from Aalto university's `Python for scientific computing <https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/#what-is-a-data-format>`_.

.. list-table::
   :header-rows: 1

   * - | Name:
     - | Human
       | readable:
     - | Space
       | efficiency:
     - | Arbitrary
       | data:
     - | Tidy
       | data:
     - | Array
       | data:
     - | Long term
       | storage/sharing:

   * - :ref:`Pickle <pickle>`
     - ❌
     - 🟨
     - ✅
     - 🟨
     - 🟨
     - ❌

   * - :ref:`CSV <csv>`
     - ✅
     - ❌
     - ❌
     - ✅
     - 🟨
     - ✅

   * - :ref:`Feather <feather>`
     - ❌
     - ✅
     - ❌
     - ✅
     - ❌
     - ❌

   * - :ref:`Parquet <parquet>`
     - ❌
     - ✅
     - 🟨
     - ✅
     - 🟨
     - ✅

   * - :ref:`npy <npy>`
     - ❌
     - 🟨
     - ❌
     - ❌
     - ✅
     - ❌

   * - :ref:`HDF5 <hdf5>`
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅

   * - :ref:`NetCDF4 <netcdf4>`
     - ❌
     - ✅
     - ❌
     - ❌
     - ✅
     - ✅

   * - :ref:`JSON <json>`
     - ✅
     - ❌
     - 🟨
     - ❌
     - ❌
     - ✅

   * - :ref:`Excel <excel>`
     - ❌
     - ❌
     - ❌
     - 🟨
     - ❌
     - 🟨

   * - :ref:`Graph formats <graph>`
     - 🟨
     - 🟨
     - ❌
     - ❌
     - ❌
     - ✅


Data Structures for ML/DL
-------------------------


ML (and DL) models require numerical input, so we must collect adaquate numerical data before training.
For ML tasks, multimedia data like image, audio, or video formats should be converted into tabular data or numerical arrays that ML models can process.
This conversion enables models to extract meaningful features, such as pixel intensities, audio frequencies or motion patterns, for tasks like classification or prediction.


Numerical array 
^^^^^^^^^^^^^^^

Numerical array is a collection of numbers arranged in a specific structure that one can perform mathematical operations on. Examples of numerical arrays are scalar (0D), row or column vector (1D), matrix (2D), and tensor (3D), *etc.*

Python offers powerful libraries like NumPy, PyTorch, TensorFlow, and Dask (parallel Numpy) to work with numerical arrays (0D to *n*D).

.. code-block:: python

   import numpy as np

   # 0D (Scalar)
   scalar = np.array(5)  

   # 1D (Vector)
   vector = np.array([1, 2, 3])  

   # 2D (Matrix)
   matrix_2D = np.array([[1, 2], [3, 4]])  

   # 3D (Matrix)
   matrix_3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
   print(matrix_3D.shape)


Tensor
^^^^^^

In ML and DL, a tensor is a mathematical object used to represent and manipulate multidimensional data. It generalizes scalars, vectors, and matrices to higher dimensions, serving as the fundamental data structure in frameworks like TensorFlow and PyTorch.

Why to use tensors in ML/DL (advantages of Tensor)?

- Generalization of scalars/vectors/matrices: Tensors extend these concepts to any number of dimensions, which is essential for handling complex data like images (3D) and videos (4D+).
- Consistency: Tensors unify data structures across ML/DL frameworks, simplifying model building, training, and deployment.
- Efficient computation: Frameworks like TensorFlow and PyTorch optimize tensor operations for speed (using GPUs/TPUs).
- Neural network representations: Input data (images, text) is converted to tensors.
- Automatic differentiation: Tensors support gradient tracking, which is vital for backpropagation in neural networks.


`HERE <>`_ we provide a tutorial about Tensor including

- Tensor creation
- Tensor's properties (`shape`, `dtype`, `ndim`)
- Tensor operations

   - indexing, slicing, transposing
   - element-wise operations: addition, subtraction, *etc.*
   - matrix multiplication(`np.dot`, `torch.matmul`)
   - reshaping, flattening, squeezing, unsqueezing
   - reduction operations: sum, mean, max along axes
   - broadcasting: Rules and examples

- Tensors in DL frameworks

   - moving tensors between CPUs and GPUs (suppose that you can access to GPU cards)


Data Preprocessing
------------------


With the huge amount of data at disposal, more and more researchers and industry professionals are finding ways to use this data for research and commercial benefits. However, most of the data available by default is too raw. It is important to preprocess it before it can be used to identify important patterns or can be used to train statistical models that can be used to make predictions.

Data preprocessing refers to steps taken to integrate, clean, transform, and organize raw data into a format that can be effectively used by ML algorithms. It’s one of the most critical steps in the ML workflow because high-quality data leads to better model performance. 

`HERE <>`_ we provide a tutorial addressing representative steps for preprocessing data in `penguins datasets <https://inria.github.io/scikit-learn-mooc/python_scripts/trees_dataset.html>`_ using Python libraries like `numpy`, `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.


Feature Engineering
-------------------


Feature engineering is a part of the broader data processing pipeline in ML workflows. It involves using domain knowledge to select, modify, or create new features -- variables or attributes -- from existing data to help algorithms better understand patterns and relationships.

Feature engineering is crucial because the quality of features directly impacts a model's predictive power. Well-crafted features can simplify complex patterns, reduce overfitting, and improve model interpretability, leading to better generalization and performance on unseen data. By tailoring features to the problem at hand, feature engineering bridges the gap between raw data and actionable insights, often making the difference between a mediocre and a high-performing model.

Feature engineering is closely related to data processing, but they serve different purposes.

- Data processing (or data preprocessing) is about cleaning and preparing data -- handling missing values, removing duplicates, correcting data types, and ensuring consistency. This step makes the data **usable**.
- Feature engineering, on the other hand, comes after basic processing and focuses on improving the predictive power of dataset.
- In essence, **data processing ensures data quality**, while **feature engineering enhances data value** for ML models.
- Both are essential steps in building effective and accurate predictive systems.
