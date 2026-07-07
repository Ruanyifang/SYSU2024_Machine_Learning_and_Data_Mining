import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from id3 import Id3Estimator
from C45 import C45Classifier

# 数据加载与预处理
data = pd.read_csv('winequality-red.csv', sep=';')
X = data.drop('quality', axis=1)
y = (data['quality'] >= 6).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 参数优化
max_depth_values = [3, 5, 7, None]

print("=== 参数优化结果 ===")
for max_depth in max_depth_values:
    print(f"\n--- max_depth = {max_depth} ---")

    # CART 优化
    cart_tree = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=max_depth)
    cart_tree.fit(X_train, y_train)
    cart_pred = cart_tree.predict(X_test)
    print("\nCART Decision Tree Performance")
    print("Accuracy:", accuracy_score(y_test, cart_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, cart_pred))

    # ID3 优化
    id3_tree = Id3Estimator(max_depth=max_depth if max_depth is not None else float('inf'))
    id3_tree.fit(X_train, y_train)
    id3_pred = id3_tree.predict(X_test)
    print("\nID3 Decision Tree Performance")
    print("Accuracy:", accuracy_score(y_test, id3_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, id3_pred))

    # C45 优化
    c45_tree = C45Classifier(max_depth=max_depth)  
    c45_tree.fit(X_train, y_train)
    c45_pred = c45_tree.predict(X_test)
    print("\nC4.5 Decision Tree Performance")
    print("Accuracy:", accuracy_score(y_test, c45_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, c45_pred))
