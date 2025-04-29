import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess_data(transaction_path, identity_path):
    # Load data
    transactions = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    
    # Merge datasets
    data = transactions.merge(identity, on='TransactionID', how='left')
    
    # Select relevant features (simplified for undergraduate level)
    features = ['TransactionAmt', 'ProductCD', 'card1', 'card4', 'addr1', 'dist1', 'DeviceType']
    target = 'isFraud'
    
    # Handle missing values
    data = data[features + [target]].copy()
    data.fillna(data.mean(numeric_only=True), inplace=True)
    data.fillna('unknown', inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in data.select_dtypes(include='object').columns:
        data[col] = le.fit_transform(data[col])
    
    # Scale numerical features
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    
    return data[features], data[target]

if __name__ == "__main__":
    X, y = load_and_preprocess_data('data/train_transaction.csv', 'data/train_identity.csv')
    print(X.head())
    print(y.value_counts())