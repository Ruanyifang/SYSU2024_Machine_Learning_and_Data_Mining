import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 从外部链接加载数据集
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]  # 自定义列名
data = pd.read_csv(url, header=None, names=columns)

# 删除不需要的ID列，将Diagnosis列转换为二进制值 (M=1, B=0)
data.drop('ID', axis=1, inplace=True)
data['Diagnosis'] = data['Diagnosis'].map({'M': 1, 'B': 0})

# 分离特征和目标
X = data.iloc[:, 1:].values  # 所有特征列
y = data.iloc[:, 0].values   # 目标列（Diagnosis）

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征值以提高模型性能
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 使用训练集拟合标准化器
X_test = scaler.transform(X_test)        # 使用相同的标准化器转换测试集

# SVM模型初始化 - 线性核
# 超参数选择：C（惩罚参数），设定为1.0
# C值越小，正则化强度越大，可能导致欠拟合；C值越大，正则化强度越小，可能导致过拟合
C_linear = 1.0
linear_svm = SVC(kernel='linear', C=C_linear, random_state=42)  # 初始化线性核SVM
linear_svm.fit(X_train, y_train)  # 训练模型

# SVM模型初始化 - 高斯核（RBF）
# 超参数选择：C（惩罚参数），设定为1.0；gamma（核函数参数），设定为'auto'
# gamma越大，决策边界越复杂，可能导致过拟合；gamma越小，决策边界越简单，可能导致欠拟合
C_rbf = 1.0
gamma_rbf = 'scale'  # 使用'scale'，表示根据特征的标准差自动调整
rbf_svm = SVC(kernel='rbf', C=C_rbf, gamma=gamma_rbf, random_state=42)  # 初始化高斯核SVM
rbf_svm.fit(X_train, y_train)  # 训练模型

# 对测试集进行预测
y_pred_linear = linear_svm.predict(X_test)
y_pred_rbf = rbf_svm.predict(X_test)

# 评估模型性能
def evaluate_model(y_test, y_pred, kernel_name):
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{kernel_name} SVM Metrics:")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")

# 评估线性核SVM
evaluate_model(y_test, y_pred_linear, 'Linear Kernel')

# 评估高斯核SVM
evaluate_model(y_test, y_pred_rbf, 'RBF Kernel')
