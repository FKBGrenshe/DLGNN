# 这个文件负责数据的加载、预处理、标准化、划分训练集和测试集
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """加载数据"""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """数据预处理：特征和标签的分离，标准化"""
    features = data.drop('label', axis=1).values
    labels = data['label'].values
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    return features_scaled, labels

def split_data(features, labels, test_size=0.3):
    """划分训练集和测试集"""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test
