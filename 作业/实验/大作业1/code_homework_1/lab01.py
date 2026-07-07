import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 数据准备
def load_data():
    # 加载乳腺癌数据集
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/breast-cancer-wisconsin.data'
    # 定义数据列名
    col_names = ['ID', 'ClumpThickness', 'UniformityOfCellSize', 'UniformityOfCellShape', 
                 'MarginalAdhesion', 'SingleEpithelialCellSize', 'BareNuclei', 
                 'BlandChromatin', 'NormalNucleoli', 'Mitoses', 'Class']
    # 读取数据
    data = pd.read_csv(url, header=None, names=col_names)
    
    # 数据预处理：替换缺失值，并删除缺失值所在的行
    data = data.replace('?', np.nan).dropna()

    # 将 'BareNuclei' 列转换为整数类型
    data['BareNuclei'] = data['BareNuclei'].astype(int)

    # 将目标变量 'Class' 映射为 0 和 1，2 表示良性，4 表示恶性
    data['Class'] = data['Class'].map({2: 0, 4: 1})

    # 特征和标签分开
    X = data.drop(['ID', 'Class'], axis=1).values  # 特征矩阵
    y = data['Class'].values  # 标签数组
    
    # 将数据集拆分为训练集和测试集，80% 用于训练，20% 用于测试
    return train_test_split(X, y, test_size=0.2, random_state=42)

# 线性分类器
class LinearClassifier:
    def __init__(self, loss='hinge', learning_rate=0.01, n_epochs=1000):
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None

    def fit(self, X, y):
        num_samples, num_features = X.shape  # 获取样本数和特征数
        # 初始化权重为零
        self.weights = np.zeros(num_features)
        
        for epoch in range(self.n_epochs):
            for i in range(num_samples):
                xi = X[i]  # 当前样本的特征
                yi = y[i]  # 当前样本的标签
                # 计算预测值
                prediction = np.dot(xi, self.weights)  # 线性组合

                if self.loss == 'hinge':
                    # Hinge Loss的实现
                    # 如果预测的结果不满足条件，则更新权重
                    if yi * prediction < 1:
                        self.weights += self.learning_rate * (yi * xi - 2 * self.weights)
                    else:
                        self.weights -= self.learning_rate * 2 * self.weights
                
                elif self.loss == 'cross_entropy':
                    # Cross-Entropy Loss的实现
                    prob = 1 / (1 + np.exp(-prediction))  # Sigmoid函数得到概率
                    gradient = prob - yi  # 计算梯度
                    self.weights -= self.learning_rate * gradient * xi  # 更新权重

    def predict(self, X):
        predictions = np.dot(X, self.weights)
        return (predictions > 0).astype(int)  # 将预测结果转换为0或1

# 评估模型性能
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return accuracy, precision, recall, f1

# 主程序
X_train, X_test, y_train, y_test = load_data()  # 加载并拆分数据

# 使用Hinge Loss训练
hinge_classifier = LinearClassifier(loss='hinge', learning_rate=0.01, n_epochs=1000)
hinge_classifier.fit(X_train, y_train)
y_pred_hinge = hinge_classifier.predict(X_test)

# 使用Cross-Entropy Loss训练
cross_entropy_classifier = LinearClassifier(loss='cross_entropy', learning_rate=0.01, n_epochs=1000)
cross_entropy_classifier.fit(X_train, y_train)
y_pred_cross_entropy = cross_entropy_classifier.predict(X_test)

# 评估模型
metrics_hinge = evaluate_model(y_test, y_pred_hinge)
metrics_cross_entropy = evaluate_model(y_test, y_pred_cross_entropy)

# 输出结果
print("Hinge Loss Metrics:")
print(f"Accuracy: {metrics_hinge[0]:.4f}, Precision: {metrics_hinge[1]:.4f}, Recall: {metrics_hinge[2]:.4f}, F1 Score: {metrics_hinge[3]:.4f}")
print("\nCross-Entropy Loss Metrics:")
print(f"Accuracy: {metrics_cross_entropy[0]:.4f}, Precision: {metrics_cross_entropy[1]:.4f}, Recall: {metrics_cross_entropy[2]:.4f}, F1 Score: {metrics_cross_entropy[3]:.4f}")
