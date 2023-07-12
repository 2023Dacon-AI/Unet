import numpy as np
import matplotlib.pyplot as plt


import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('submit.csv')

# 데이터프레임 확인
#print(df.head())

test = df['mask_rle'][0]

#print(test)

plt.figure(figsize=(10, 10))
plt.imshow(str(test), cmap='gray')
plt.axis('off')
plt.show()