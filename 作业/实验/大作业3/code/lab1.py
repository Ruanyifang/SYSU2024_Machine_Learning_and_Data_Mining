import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 1. 数据加载与预处理
# 加载数据集
data = pd.read_csv('winequality-red.csv', sep=';')

# 检查数据基本信息
print(data.info())
print(data.describe())

# 分离特征和标签
X = data.drop('quality', axis=1)
y = data['quality']

# 标准化处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. PCA降维
# 定义不同的主成分数量
n_components_list = [2, 3, 5]

pca_results = {}
explained_variances = {}

for n in n_components_list:
    pca = PCA(n_components=n)
    X_pca = pca.fit_transform(X_scaled)
    pca_results[n] = X_pca
    explained_variances[n] = pca.explained_variance_ratio_
    print(f'主成分数量: {n}')
    print(f'解释方差比例: {explained_variances[n]}')

# 3. 可视化分析
plt.figure(figsize=(8, 6))
plt.scatter(pca_results[2][:, 0], pca_results[2][:, 1], c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('Principal Components 1')  # 主成分1
plt.ylabel('Principal Components 2')  # 主成分2
plt.title('Feature space distribution after PCA dimensionality reduction')  # PCA降维后特征空间分布
plt.colorbar(label='Quality Rating')  # 质量评分
plt.show()


# 累计解释方差比例
plt.figure(figsize=(8, 6))
components = np.arange(1, len(pca.explained_variance_ratio_) + 1)
plt.bar(components, explained_variances[5], alpha=0.5, align='center',
        label='Individual explained variance')
plt.step(components, np.cumsum(explained_variances[5]), where='mid',
         label='Cumulative explained variance')
plt.xlabel('Number of principal components')  # 主成分数量
plt.ylabel('Proportion of explained variance')  # 解释方差比例
plt.title('PCA explained variance proportion')  # PCA解释方差比例
plt.legend(loc='best')
plt.show()
