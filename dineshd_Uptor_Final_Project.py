import numpy as np
import pandas as pd
# from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the Boston housing dataset
boston = pd.read_csv("boston_house_prices.csv")
print(boston.columns)

# X = boston.data
# y = boston.target
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Apply PCA to reduce dimensionality
# n_components = 3  # Choose the number of principal components
# pca = PCA(n_components=n_components)
# X_train_pca = pca.fit_transform(X_train)
# X_test_pca = pca.transform(X_test)
#
# # Train a linear regression model on the PCA-transformed data
# model = LinearRegression()
# model.fit(X_train_pca, y_train)
#
# # Make predictions on the test set
# y_pred = model.predict(X_test_pca)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse}")
#
# # Optionally, visualize the results (if n_components is 1, 2, or 3)
# if n_components == 1:
#     plt.scatter(X_test_pca, y_test, label='Actual')
#     plt.scatter(X_test_pca, y_pred, label='Predicted')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Target Variable (Median House Value)')
#     plt.legend()
#     plt.title('PCA + Linear Regression (1 Component) - Boston Dataset')
#     plt.show()
#
# elif n_components == 2:
#     plt.scatter(X_test_pca[:, 0], y_test, label='Actual')
#     plt.scatter(X_test_pca[:, 0], y_pred, label='Predicted')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Target Variable (Median House Value)')
#     plt.legend()
#     plt.title('PCA + Linear Regression (2 Components) - Boston Dataset')
#     plt.show()
#
#     plt.scatter(X_test_pca[:, 1], y_test, label='Actual')
#     plt.scatter(X_test_pca[:, 1], y_pred, label='Predicted')
#     plt.xlabel('Principal Component 2')
#     plt.ylabel('Target Variable (Median House Value)')
#     plt.legend()
#     plt.title('PCA + Linear Regression (2 Components) - Boston Dataset')
#     plt.show()
#
# elif n_components == 3:
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_test, marker='o', label='Actual')
#     ax.scatter(X_test_pca[:, 0], X_test_pca[:, 1], X_test_pca[:, 2], c=y_pred, marker='x', label='Predicted')
#     ax.set_xlabel('Principal Component 1')
#     ax.set_ylabel('Principal Component 2')
#     ax.set_zlabel('Principal Component 3')
#     ax.legend()
#     ax.set_title("PCA + Linear Regression (3 Components) - Boston Dataset")
#     plt.show()
#
# # Demonstrate the explained variance ratio
# print("Explained Variance Ratio:", pca.explained_variance_ratio_)