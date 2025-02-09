import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load Iris dataset from CSV
df = pd.read_csv("iris.data.csv")

# Histogram
df.hist(figsize=(10, 6))
plt.suptitle("Histogram of Iris Features")
plt.show()

# Stem-and-leaf plot (approximation using dot plot)
for col in df.columns[:-1]:  # Exclude target column if present
    print(f"Stem-and-leaf plot for {col}:")
    values = np.round(df[col] * 10).astype(int)
    stems = np.unique(values // 10)
    for stem in stems:
        leaves = values[values // 10 == stem] % 10
        print(f"{stem} | {' '.join(map(str, sorted(leaves)))}")
    print()

# Box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df.iloc[:, :-1])  # Exclude target column if present
plt.title("Boxplot of Iris Features")
plt.xticks(rotation=45)
plt.show()

# Scatter plot
sns.pairplot(df.iloc[:, :-1])  # Exclude target column if present
plt.suptitle("Scatter Plot Matrix of Iris Features", y=1.02)
plt.show()
