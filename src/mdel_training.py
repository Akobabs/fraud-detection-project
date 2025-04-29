import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import shap
import matplotlib.pyplot as plt
from data_preprocessing import load_and_preprocess_data

def train_and_evaluate():
    # Load data
    X, y = load_and_preprocess_data('data/train_transaction.csv', 'data/train_identity.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred)
    }
    
    # SHAP explanations
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # Save SHAP summary plot
    shap.summary_plot(shap_values[1], X_test, feature_names=X.columns, show=False)
    plt.savefig('figures/shap_summary.png')
    plt.close()
    
    return model, explainer, metrics

if __name__ == "__main__":
    model, explainer, metrics = train_and_evaluate()
    print("Model Performance:", metrics)