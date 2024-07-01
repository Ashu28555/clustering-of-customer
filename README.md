# clustering-of-customer
# Task 2 : Create a kmeans algorithm to group customers of a retail store based on their purchase history.

#impoting Libraries
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("Mall_Customers.csv")

data

data.shape

data.info()

data.describe()

# Select the features for clustering
x = data[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

scaler

x_scaled

# Determine the optimal number of clusters using the elbow method
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow method graph
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=42)
y_kmeans = kmeans.fit_predict(x_scaled)

# Add the cluster assignments back to the original dataframe
data['Cluster'] = y_kmeans

# Display the first few rows of the dataframe with cluster assignments
data

# Scatter plot to visualize the clusters
sns.scatterplot(x='Age', y='Annual Income (k$)', hue='Cluster', palette='viridis', data=data, s=100)
plt.title('Clusters by Age and Annual Income')
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')

sns.scatterplot(x='Age', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=data, s=100)
plt.title('Clusters by Age and Spending Score')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')

sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='viridis', data=data, s=100)
plt.title('Clusters by Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()

# The End

