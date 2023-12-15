import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

def plot_k_value_analysis(X_train, y_train, X_test, y_test, max_k, title):
    k_values = range(1, max_k + 1)
    accuracies = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))

    plt.plot(k_values, accuracies)
    plt.title(title)
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('Accuracy')
    plt.show()
def plot_model_accuracies(accuracies, title):
    # Sort the models by their accuracy in descending order
    sorted_accuracies = {k: v for k, v in sorted(accuracies.items(), key=lambda item: item[1], reverse=True)}

    model_names = list(sorted_accuracies.keys())
    model_scores = list(sorted_accuracies.values())

    bars = plt.bar(model_names, model_scores, color='skyblue')

    # Annotate the bars with the accuracy values
    for bar in bars:
        yval = bar.get_height()
        plt.annotate(f'{yval:.3f}',  # Format to 1 decimal place
                     (bar.get_x() + bar.get_width() / 2, yval),
                     va='bottom', ha='center',
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords='offset points')

    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.title(title)
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.show()
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def plot_decision_boundaries(X, y, model, title):
    # Setting min and max values and giving some padding
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # Predict class using data and KNN classifier
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.contourf(xx, yy, Z, alpha=0.4)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')

    # Create legend
    class_labels = np.unique(y)
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'Class {label}',
                          markerfacecolor=scatter.cmap(scatter.norm(label)), markersize=10) for label in class_labels]
    plt.legend(handles=handles)

    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()


def run_models(file_path):
    # Load the dataset
    data = pd.read_csv(file_path)

    # Adjust class labels to start from 0
    data['class'] = data['class']  # Adjust if class labels are not starting from 0

    # Determine features based on the file
    features = ['PC1', 'PC2']
    if 'PC3' in data.columns:
        features.append('PC3')

    # Splitting the data into features and target
    X = data[features]
    y = data['class']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Train and evaluate each model
    for name, model in models.items():
        print(f"Training {name} with {file_path}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        print(f"{name} Results:")
        print("Accuracy:", accuracy)
        print("Classification Report:\n", report)

        # Plot decision boundaries for KNN
        if name == 'KNN':
            plot_decision_boundaries(np.array(X), np.array(y), model, f"KNN Decision Boundary with {len(features)} PCs")
        if name == 'KNN':
            plot_confusion_matrix(y_test, y_pred, 'Confusion Matrix for KNN')
        print("------------------------------------------------------\n")
        # After running models with two PCs
        if name == 'KNN':
            plot_k_value_analysis(X_train, y_train, X_test, y_test, 20, 'K-Value Analysis for KNN with 2 PCs')

    accuracies = {}
    for name, model in models.items():
        print(f"Training {name} with {file_path}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies[name] = accuracy  # Store accuracy in the dictionary
        # ... [rest of your existing code] ...

    return accuracies


# Collecting accuracies for each dataset
accuracies_2pc = run_models('PCA_output.csv')
# Plotting accuracies for each dataset
plot_model_accuracies(accuracies_2pc, 'Model Accuracies with 2 Principal Components')
# Run models with three PCs (if applicable)
# run_models('PCA_output3.csv')



