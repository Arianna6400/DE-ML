# Data Engineering & Machine Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-1.3%2B-lightblue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24%2B-yellowgreen)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4%2B-blue)
![GitHub](https://img.shields.io/badge/GitHub-Repository-black)

This repository contains a collection of **Jupyter notebooks** developed as part of the Data Engineering & Machine Learning module. The aim is to provide foundational knowledge in information and data engineering, along with key machine learning concepts. Python is the main programming language used to enhance both mathematical and coding skills, while exploring advanced topics like modern machine learning platforms, data visualization, and deep learning.

# Index

- [Data Engineering & Machine Learning](#data-engineering--machine-learning)
- [Index](#index)
  - [📌 Main Topics](#-main-topics)
  - [📘 Activities](#-activities)
    - [1. Linear Algebra & Python](#1-linear-algebra--python)
    - [2. Probability & Python](#2-probability--python)
    - [3. Artificial Neurons](#3-artificial-neurons)
    - [4. Logistics Regression and SVMs](#4-logistics-regression-and-svms)
    - [5. Data Processing](#5-data-processing)
    - [6. Neural Networks](#6-neural-networks)
  - [👥 Contribution](#-contribution)

## 📌 Main Topics

The following aspects of machine learning are covered:

1. **Machine Learning Algorithms**: Introduction to fundamental techniques like Perceptron, Logistic Regression, Support Vector Machines, and Multi-Layer Perceptron, using libraries like *Scikit-learn*.

2. **Data Processing and Visualization**: Applying techniques for data pre-processing, post-processing, and visualization using libraries like *NumPy*, *Pandas*, and *Matplotlib*.

3. **Practical Applications**: Using real-world datasets to identify trends, make inferences, and implement solutions for tasks such as image classification.

## 📘 Activities

### 1. Linear Algebra & Python

The purpose of this activity is to become confident with *matrix addition*, *subtraction*, *multiplication*, *determining matrix determinants*, *inverse*, *cross-product* and *eigenvalues*, developing code for these operations and creating Python functions to represent this functionality in a reusable way.

The first Jupyter Notebook could be found here: [Activity 1](https://github.com/Arianna6400/DE-ML/blob/master/Activity1/Etivity1_LinearAlgebra.ipynb).

### 2. Probability & Python

This activity focuses on the application of *probability theory*, utilizing Python's *Random* module to simulate random events and generate datasets. Statistical properties of these datasets are calculated and compared to theoretical estimates to assess the alignment between simulated and expected outcomes. Additionally, data distributions are visualized to gain deeper insights into the stochastic nature of the generated datasets.

This exploration is essential for understanding the probabilistic foundations of real-world machine learning problems, where data and algorithms often exhibit stochastic behavior.

The activity also extends Python programming skills by introducing topics such as reading CSV files and object-oriented programming through the use of classes. Furthermore, it serves as an introduction to **Reinforcement Learning**, complementing previous knowledge of supervised and unsupervised learning, thus broadening the understanding of different learning paradigms in machine learning. 

The second Jupyter Notebook could be found here: [Activity 2](https://github.com/Arianna6400/DE-ML/blob/master/Activity2/Etivity2_Probability.ipynb).

### 3. Artificial Neurons

Over the course of this activity, the focus is on exploring the fundamentals of artificial neurons, also known as **Perceptrons**, which are essential components of neural networks and deep learning. The activity covers the theory of perceptron learning and applies it to a dataset that can be linearly separated. In particular, an **Adaline** perceptron algorithm will be used, which is a perceptron that is trained with a gradient descent algorithm.

This first Jupyter Notebook could be found here: [Activity 3.1](https://github.com/Arianna6400/DE-ML/blob/master/Activity3/Etivity3_Adaline.ipynb)

Additionally, the implementation of a perceptron using gradient descent in scikit-learn is examined and compared with the Perceptron learning rule. The activity also involves working with datasets using the Pandas library and further exploring the scikit-learn library.

The second Jupyter Notebook could be found here: [Activity 3.2](https://github.com/Arianna6400/DE-ML/blob/master/Activity3/Etivity3_ScikitLearn.ipynb)

Through this task, foundational concepts of machine learning, such as weights, epochs, and learning rates, are introduced and explored, providing a solid base for understanding more advanced neural network models.

### 4. Logistics Regression and SVMs 

This notebook investigates the use of *Support Vector Machines* (**SVM**) in both classical and applied classification contexts, demonstrating SVM’s versatility across three sections.

- **Part 1: Linear SVM on the Iris Dataset**

    The first section applies a linear SVM model to classify the three species in the **Iris dataset** (*Iris-setosa*, *Iris-versicolor*, and *Iris-virginica*) based on petal length and width features. Data preprocessing includes scaling and splitting into training and test sets to optimize the model's performance. The linear SVM model is trained, and its accuracy is evaluated using a confusion matrix and classification report. The decision boundary is visualized to illustrate the linear model’s effectiveness in separating the classes.

- **Part 2: Non-Linear SVM on a Randomly Generated Dataset**

    The second section demonstrates a non-linear SVM with a *radial basis function* (**RBF**) kernel on a randomly generated dataset. This section investigates how the non-linear SVM model captures complex, non-linear boundaries, showcasing its adaptability to data that is not linearly separable. The decision boundaries for this synthetic data are visualized to highlight the differences between linear and non-linear classification approaches.

- **Part 3: Network Intrusion Detection with SVM**

    The final section explores SVM’s application in **Network Intrusion Detection**, classifying network traffic as either normal or malicious. Using an intrusion detection dataset, the data is preprocessed and scaled, and both linear and non-linear SVM models are trained. Performance is assessed with metrics tailored for imbalanced data, providing insights into the SVM’s effectiveness in identifying intrusions and demonstrating its relevance in cybersecurity.

The Jupyter Notebook could be found here: [Activity 4](https://github.com/Arianna6400/DE-ML/blob/master/Activity4/Etivity4_SVM.ipynb)

### 5. Data Processing

This Jupyter notebook provides a comprehensive exploration of feature selection and dimensionality reduction techniques applied to a classification problem. The tasks include analyzing the dataset, selecting important features, and training an **SVM** classifier. The notebook also investigates the role of **PCA** and **Recursive Feature Elimination** (**RFE**) in improving model performance. Throughout the notebook, *Support Vector Machines* are used as the classifier of choice. Their robustness and interpretability make them ideal for assessing the impact of feature selection and dimensionality reduction. Key metrics, such as **accuracy** and **confusion matrices**, are used to evaluate model performance, and visualizations illustrate decision boundaries and feature importance.

The first task focuses on feature selection, a critical preprocessing step for machine learning. By carefully analyzing the dataset, various techniques are employed to identify the most relevant features for classification. The notebook begins with exploratory data analysis, visualizing distributions and gaining insights into the dataset's structure. From there, methods such as *variance thresholding* and *univariate feature selection* (`f_classif`) are used to rank features and evaluate their importance. This approach ensures that the dataset is both interpretable and efficient for downstream tasks.

In the second task, the notebook explores dimensionality reduction using *Principal Component Analysis* (**PCA**). PCA is applied to project the data into a lower-dimensional space while retaining the variance necessary for accurate predictions. This section includes visualizations that explain how much variance is preserved with each principal component, providing a clear picture of the trade-off between dimensionality and information retention. A *Support Vector Machine* (**SVM**) is then trained on the PCA-transformed data, showcasing how dimensionality reduction can simplify models without compromising accuracy.

The final task introduces *Recursive Feature Elimination* (**RFE**) as a systematic method to identify the most critical features. By iteratively removing the least important features and retraining the model, RFE highlights how feature selection can enhance model performance. The notebook provides detailed analyses and comparisons, examining how the number of features impacts classification results.

The Jupyter Notebook could be found here: [Activity 5](https://github.com/Arianna6400/DE-ML/blob/master/Activity5/Etivity5.ipynb)

### 6. Neural Networks

This notebook digs into the use of *Perceptrons* and *Multi-Layer Perceptrons* (**MLPs**) to classify the **Fashion-MNIST** dataset. Fashion-MNIST is a dataset of $70,000$ grayscale images, each belonging to one of ten clothing categories such as T-shirts, trousers, and dresses. The aim is to explore various neural network architectures, analyze their performance, and gain insights into training behaviors, overfitting, and generalization.

The tasks are structured to provide a comprehensive journey through different depths and widths of neural networks. Starting from a basic Perceptron, the notebook progressively builds up to deeper and more complex networks, analyzing how each architecture impacts classification accuracy and computational efficiency.

- **Dataset Overview and Visualization**

  The Fashion-MNIST dataset is loaded and preprocessed for the experiments. Each image, represented as a $28x28$ grayscale pixel array, is flattened into a $784$-dimensional vector for input into the neural networks. An initial visualization showcases sample images alongside their respective labels, providing a tangible sense of the dataset's structure and variety. Data is then split into training and testing sets, followed by standardization to improve optimization performance during training.

- **Task 1: Training a Perceptron**

  The notebook begins by implementing a simple Perceptron, a linear classifier, to classify the Fashion-MNIST dataset. Despite being a basic model, the Perceptron achieves an accuracy of approximately $81.66\%$, which, while promising, highlights its limitations in capturing the complexities of the dataset.

  An analysis of misclassified samples reveals the challenges in distinguishing similar categories, such as shirts and pullovers. These observations provide a foundation for transitioning to more sophisticated models.

- **Task 2: Single-Layer Multi-Layer Perceptrons (MLPs)**

  Building on the Perceptron, an MLP with a single hidden layer containing $20$ neurons is trained. This architecture allows the model to learn non-linear relationships, achieving higher accuracy than the Perceptron. By examining training and testing accuracies, overfitting is identified as a potential issue, emphasizing the need for careful regularization and architectural tuning.

  To better understand the impact of hidden layer width, experiments are conducted with varying neuron counts. Results indicate diminishing returns in accuracy as the number of neurons increases, with a sweet spot around $50$ neurons. This finding highlights the importance of balancing model complexity with generalization ability.

- **Task 3: Exploring Deeper Networks**

  The notebook transitions to experimenting with deeper network architectures, stacking multiple hidden layers with different configurations. Depth introduces hierarchical feature extraction, enabling the model to better capture intricate patterns in the data. For instance, a network with three hidden layers ($100, 100, 50$ neurons) achieves comparable performance to a very wide single-layer network, but with significantly fewer parameters.

  The comparative analysis demonstrates the efficiency of deeper architectures, particularly in terms of memory and computational cost. The discussion also underscores the advantages of depth in generalization over wide architectures that may overfit or underperform on complex datasets.

- **Task 4: Evaluating Model Performance**

  A confusion matrix is generated for the best-performing MLP architecture, offering detailed insights into category-wise predictions. Misclassifications are analyzed to identify patterns and weaknesses in the model's learning. For example, categories with visual similarities, such as shirts and pullovers, exhibit higher confusion rates.

The Jupyter Notebook could be found here: [Activity 6](https://github.com/Arianna6400/DE-ML/blob/master/Activity6/Etivity6.ipynb)

## 👥 Contribution

|Nome | GitHub |
|-----------|--------|
| 👩 `Agresta Arianna` | [Click here](https://github.com/Arianna6400) |

