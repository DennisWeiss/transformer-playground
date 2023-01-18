import pandas as pd


df = pd.read_csv('data7.csv')
model_name = 'DeepDataDepthAnomalyDetector'
sub_dataset = 'CIFAR10'
# model_name = 'MostNaive'

df = df[df[model_name].notna()]

print(round(10000 * df[model_name].mean()) / 100)

print(df[df['params'].str.contains(sub_dataset)].sort_values(by='params'))
print(round(10000 * df[df['params'].str.contains(sub_dataset)][model_name].mean()) / 100)