import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

df=pd.read_csv('Walmart_sales.csv',delimiter=',')

print(df.head)
print(df.tail)
print(df.shape)
print(df.describe)

df.plot.scatter(df)