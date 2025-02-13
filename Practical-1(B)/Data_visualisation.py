import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import stemgraphic
from mpl_toolkits.mplot3d import Axes3D

file_path='Practical-1(B)/housing.csv'
df=pd.read_csv(file_path)
# print (df);
#dropping unnecessary colums
df_numeric = df.drop(columns=['ocean_proximity'])

#  1. Histogram
plt.figure(figsize=(12,6))
df_numeric.hist(bins=15, edgecolor='black', figsize=(10,6))
plt.suptitle("Histograms of Clifornia Dataset Features")
plt.show()

#  2. Stem-and-Leaf Plot
stemgraphic.stem_graphic(df["median_income"])
plt.show()

#  3. Box Plot
plt.figure(figsize=(10,6))
sns.boxplot(data=df.drop(columns=['ocean_proximity']))
plt.title("Box Plot of ocean_proximity Features")
plt.show()

 # 4. Pie Chart
area = df['ocean_proximity'].value_counts()
plt.figure(figsize=(6,6))
plt.pie(area, labels=area.index, autopct='%1.1f%%', colors=['red',
'blue', 'green','orange'])
plt.title("land Distribution in Iris Dataset")
plt.show()

# Scatter Plot
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='longitude', y='latitude', hue='ocean_proximity')
plt.title("Scatter Plot of longitude vs latitude")
plt.xlabel("Latitude (in deg)")
plt.ylabel("Longitude (in deg)")
plt.show()

#  6. Contour Plot
plt.figure(figsize=(8,6))
sns.kdeplot(data=df, x='longitude', y='latitude', hue='ocean_proximity', fill=True)
plt.title("Contour Plot of longitude vs. latitude")
plt.show()