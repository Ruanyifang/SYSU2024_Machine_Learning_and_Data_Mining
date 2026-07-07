import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from id3 import Id3Estimator
from id3 import export_text as export_text_id3
from C45 import C45Classifier

# 1. 数据加载与预处理
# 加载数据集
data = pd.read_csv('winequality-red.csv', sep=';')

# 分离特征和标签
X = data.drop('quality', axis=1)
y = data['quality']

# 将标签转为二分类问题（质量>=6为好酒，<6为普通酒）
y = (y >= 6).astype(int)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. 构建决策树模型
# 2.1 CART (sklearn的默认算法)
cart_tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=5)
cart_tree.fit(X_train, y_train)
cart_pred = cart_tree.predict(X_test)

# 2.2 ID3
id3_tree = Id3Estimator(max_depth=5)
id3_tree.fit(X_train, y_train)
id3_pred = id3_tree.predict(X_test)

# 2.3 C4.5
c45_tree = C45Classifier()  # 初始化 C4.5 分类器
c45_tree.fit(X_train, y_train)  # 使用训练数据训练模型
c45_pred = c45_tree.predict(X_test)  # 对测试集进行预测

# 3. 模型性能评估
# 3.1 CART
print("CART Decision Tree Performance")
print("Accuracy:", accuracy_score(y_test, cart_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, cart_pred))
print("Classification Report:\n", classification_report(y_test, cart_pred))

# 3.2 ID3
print("\nID3 Decision Tree Performance")
print("Accuracy:", accuracy_score(y_test, id3_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, id3_pred))
print("Classification Report:\n", classification_report(y_test, id3_pred))

# 3.3 C4.5
print("\nC4.5 Decision Tree Performance")
print("Accuracy:", accuracy_score(y_test, c45_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, c45_pred))
print("Classification Report:\n", classification_report(y_test, c45_pred))

# 4. 树结构输出
print("\nCART Tree Structure:")
print(export_text(cart_tree, feature_names=list(X.columns)))

print("\nID3 Tree Structure:")
print(export_text_id3(id3_tree.tree_, feature_names=list(X.columns)))
