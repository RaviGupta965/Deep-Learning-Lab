import numpy as np
import pandas as pd;

file_path='Practical-1(B)/housing.csv'
df=pd.read_csv(file_path)
# print (df);
#dropping unnecessary colums
df_numeric = df.drop(columns=['ocean_proximity'])

# 1. Mean,Median Mode
mean_value=df_numeric.mean()
print('Mean: \n',mean_value)
print('\n');
median_value=df_numeric.median()
print('Median: \n',median_value)
print('\n')
mode_value=df_numeric.mode().iloc[0]
print('Mode: \n',mode_value)

# Percentile
print('\n');
percentiles = {
"25th Percentile": df_numeric.quantile(0.25),
"50th Percentile": df_numeric.quantile(0.50),
"75th Percentile": df_numeric.quantile(0.75)
}
print(percentiles)

# 3. Range
range_values = df_numeric.max() - df_numeric.min()
print("Range:\n", range_values)

# 4. Variance
variance_values = df_numeric.var()
print("\nVariance:\n", variance_values)

# 5. Standard Deviation
std_dev_values = df_numeric.std()
print("\nStandard Deviation:\n", std_dev_values)

# 6. Correlation Matrix
correlation_matrix = df_numeric.corr()
print("\nCorrelation Matrix:\n", correlation_matrix)

# 7. Covariance Matrix
covariance_matrix = df_numeric.cov()
print("\nCovariance Matrix:\n", covariance_matrix)