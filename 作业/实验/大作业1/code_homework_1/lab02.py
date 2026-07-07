import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载乳腺癌数据集
data = load_breast_cancer()
X = data.data
y = data.target

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# 定义一个函数来生成多项式特征并训练模型
def polynomial_regression(degree):
    # 多项式特征生成
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # 使用逻辑回归作为分类器
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train_poly, y_train)
    
    # 预测并计算性能指标
    y_pred_train = model.predict(X_train_poly)
    y_pred_test = model.predict(X_test_poly)
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test)
    recall = recall_score(y_test, y_pred_test)
    f1 = f1_score(y_test, y_pred_test)
    
    return train_acc, test_acc, precision, recall, f1

# 比较不同多项式次数下的模型表现
degrees = [1, 2, 3, 4, 5]
train_accs = []
test_accs = []
precisions = []
recalls = []
f1_scores = []

for degree in degrees:
    train_acc, test_acc, precision, recall, f1 = polynomial_regression(degree)
    train_accs.append(train_acc)
    test_accs.append(test_acc)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    print(f"Degree: {degree}")
    print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
    print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}")
    print("-" * 40)

# 可视化不同degree下的模型准确率
plt.plot(degrees, train_accs, label="Train Accuracy")
plt.plot(degrees, test_accs, label="Test Accuracy")
plt.xlabel("Polynomial Degree")
plt.ylabel("Accuracy")
plt.legend()
plt.title("Train vs Test Accuracy for Different Polynomial Degrees")
plt.show()
