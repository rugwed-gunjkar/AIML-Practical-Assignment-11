# AIML Practical Assignment No. 11

## Title
Implement K-Means Clustering Algorithm on a Multidimensional Dataset from UCI Repository and Analyse Performance Using Silhouette Score and Cluster Visualization.

## Objective
To implement the K-Means clustering algorithm on the Wine dataset, perform data standardization, reduce dimensions using PCA, and evaluate clustering performance using the Silhouette Score.

## Tools and Technologies
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- VS Code

## Dataset
The Wine dataset is used in this experiment.

Dataset details:
- 178 samples
- 13 numerical features
- 3 actual wine classes

## Algorithm Used

### K-Means Clustering
K-Means is an unsupervised machine learning algorithm that groups similar data points into a specified number of clusters.

### Principal Component Analysis (PCA)
PCA is used to reduce the 13-dimensional Wine dataset into two dimensions for visualization.

## Data Preprocessing
StandardScaler is used to standardize the features before applying K-Means clustering.

## Evaluation Metric
The Silhouette Score is used to evaluate the quality of clustering. It measures how well each data point fits within its assigned cluster compared to other clusters.

## Result
K-Means clustering was successfully applied to the standardized Wine dataset. PCA was used to reduce the dimensions to two, and the resulting clusters were visualized using a scatter plot.

## Conclusion
In this experiment, we successfully implemented K-Means clustering on the Wine dataset. The data was standardized, and PCA was used for dimensionality reduction and visualization. The Silhouette Score helped evaluate cluster separation and clustering quality.