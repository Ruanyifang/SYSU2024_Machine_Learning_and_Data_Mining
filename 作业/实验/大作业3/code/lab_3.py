import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from id3 import Id3Estimator

# 1. 数据加载与预处理
# 加载数据集
data = pd.read_csv('winequality-red.csv', sep=';')

# 分离特征和标签
X = data.drop('quality', axis=1)
y = data['quality']

# 将标签转为二分类问题（质量>=6为好酒，<6为普通酒）
y = (y >= 6).astype(int)

# 标准化特征数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 2. PCA 降维与模型训练
def evaluate_pca_with_id3(n_components):
    print(f"\n--- PCA with {n_components} components ---")

    # PCA 降维
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # 构建 ID3 决策树模型
    id3_tree = Id3Estimator(max_depth=7)  # 限制最大深度为7
    id3_tree.fit(X_train_pca, y_train)
    y_pred = id3_tree.predict(X_test_pca)

    # 评估模型性能
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)

    # 打印结果
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

# 3. 对比不同主成分数量下的性能
for n_components in [2, 3, 5]:  # 实验不同的主成分数量
    evaluate_pca_with_id3(n_components)

