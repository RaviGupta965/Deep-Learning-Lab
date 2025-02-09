from sklearn import datasets
import pandas as pd

# Load Iris dataset
iris = pd.read_csv('iris.data.csv')

# Compute summary statistics
summary = iris.describe()

# Display summary statistics
print(summary)