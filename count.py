import pandas as pd
from sklearn.model_selection import train_test_split

# Giả sử dữ liệu ban đầu có cột 'text' và 'label'
data = pd.read_csv('Data/combined_spam_dataset.csv')

# Tách dữ liệu thành training, testing và validation
X_train, X_temp, y_train, y_temp = train_test_split(data['text'], data['label'], test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Kiểm tra số lượng spam và ham ở mỗi tập
print("Training data counts:")
print(y_train.value_counts())

print("\nValidation data counts:")
print(y_val.value_counts())

print("\nTesting data counts:")
print(y_test.value_counts())