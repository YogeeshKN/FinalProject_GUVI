import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load the Toxic Tweets dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    return df

# Split the dataset into train and test sets
def split_dataset(df):
    X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['Toxicity'], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Convert text to Bag of Words or TF-IDF
def convert_text_features(X_train, X_test, method='bow'):
    if method == 'bow':
        vectorizer = CountVectorizer()
    elif method == 'tfidf':
        vectorizer = TfidfVectorizer()
    else:
        raise ValueError("Invalid method. Use 'bow' or 'tfidf'.")

    X_train_transformed = vectorizer.fit_transform(X_train)
    X_test_transformed = vectorizer.transform(X_test)

    return X_train_transformed, X_test_transformed

# Train and evaluate classifiers
def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    return precision, recall, f1, cm, roc_auc, y_pred

# Plot ROC-AUC curve
def plot_roc_curve(y_test, y_pred, title):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title + ' - ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Main function
def main():
    file_path = 'E:\GUVI Projects\Toxic Tweets Dataset\FinalBalancedDataset.csv'
    df = load_dataset(file_path)

    X_train, X_test, y_train, y_test = split_dataset(df)

    # Experiment with Bag of Words and TF-IDF
    for method in ['bow', 'tfidf']:
        print(f"\nExperimenting with {method.capitalize()} features:")
        X_train_transformed, X_test_transformed = convert_text_features(X_train, X_test, method)

        classifiers = {
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'Naive Bayes': MultinomialNB(),
            'K-NN': KNeighborsClassifier(),
            'SVM': SVC()
        }

        for name, classifier in classifiers.items():
            print(f"\nResults for {name} Classifier:")
            precision, recall, f1, cm, roc_auc, y_pred = train_and_evaluate_classifier(classifier, X_train_transformed, y_train, X_test_transformed, y_test)
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
            print(f"Confusion Matrix:\n{cm}")
            print(f"ROC-AUC Score: {roc_auc:.4f}")

            # Plot ROC curve for binary classifiers
            if len(np.unique(y_train)) == 2:
                plot_roc_curve(y_test, y_pred, f"{name} ({method.capitalize()})")

if __name__ == "__main__":
    main()
