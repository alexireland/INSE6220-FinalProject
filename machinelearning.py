import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

def run_cross_validation(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Adjust class labels to start from 0
    data['class'] = data['class'] - 1

    # Determine features based on the file
    features = ['PC1', 'PC2']
    if 'PC3' in data.columns:
        features.append('PC3')

    # Splitting the data into features and target
    X = data[features]
    y = data['class']

    # Define the models
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'SVM': SVC(),
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    # Stratified K-Fold cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation and print results
    for name, model in models.items():
        print(f"Cross-validating {name} with {file_path}...")
        scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
        print(f"{name} Cross-Validation Accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")
        print("------------------------------------------------------\n")

# Run cross-validation with two PCs
run_cross_validation('PCA_output.csv')

# Run cross-validation with three PCs
run_cross_validation('PCA_output3.csv')
