import pandas as pd


df = pd.read_csv('data3.csv')

df = df[df['DeepDataDepthAnomalyDetector'].notna()]



print(df[df['params'].str.contains('CIFAR10')].sort_values(by='params'))
print(df[df['params'].str.contains('CIFAR10')]['DeepDataDepthAnomalyDetector'].mean())