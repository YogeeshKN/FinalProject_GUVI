import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Load the dataset
def load_data():
    train_data1 = np.load('E:\GUVI Projects\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_train.npz')['arr_0']
    test_data1 = np.load('E:\GUVI Projects\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\X_kannada_MNIST_test.npz')['arr_0']
    train_labels = np.load('E:\GUVI Projects\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_train.npz')['arr_0']
    test_labels = np.load('E:\GUVI Projects\Kannada_MNIST_datataset_paper\Kannada_MNIST_npz\Kannada_MNIST\y_kannada_MNIST_test.npz')['arr_0']
    nsamples, nx, ny = train_data1.shape
    train_data = train_data1.reshape((nsamples,nx*ny))
    msamples, mx, my = test_data1.shape
    test_data = test_data1.reshape((msamples,mx*my))

    return train_data, test_data, train_labels, test_labels

# Apply PCA
def apply_pca(train_data, test_data, n_components):
    pca = PCA(n_components=n_components)
    train_data_pca = pca.fit_transform(train_data)
    test_data_pca = pca.transform(test_data)
    return train_data_pca, test_data_pca

# Train and evaluate classifiers
def train_and_evaluate_classifier(classifier, train_data, test_data, train_labels, test_labels):
    classifier.fit(train_data, train_labels)
    predictions = classifier.predict(test_data)

    # Metrics
    print("Classification Report:")
    print(classification_report(test_labels, predictions))

    print("\nConfusion Matrix:")
    print(confusion_matrix(test_labels, predictions))

    # Calculate ROC-AUC
    if len(np.unique(train_labels)) == 2:
        roc_auc = roc_auc_score(test_labels, predictions)
        print(f"\nROC-AUC Score: {roc_auc}")

        # Plot ROC curve for binary classification
        fpr, tpr, _ = roc_curve(test_labels, predictions)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.show()

# Main function
def main():
    # Load data
    train_data, test_data, train_labels, test_labels = load_data()

    # Perform PCA with different component sizes
    component_sizes = [10, 15, 20, 25, 30]

    for n_components in component_sizes:
        print(f"\nPCA Components: {n_components}")
        train_data_pca, test_data_pca = apply_pca(train_data, test_data, n_components)

        # Decision Tree
        print("\nDecision Tree:")
        dt_classifier = DecisionTreeClassifier()
        train_and_evaluate_classifier(dt_classifier, train_data_pca, test_data_pca, train_labels, test_labels)

        # Random Forest
        print("\nRandom Forest:")
        rf_classifier = RandomForestClassifier()
        train_and_evaluate_classifier(rf_classifier, train_data_pca, test_data_pca, train_labels, test_labels)

        # Naive Bayes
        print("\nNaive Bayes:")
        nb_classifier = GaussianNB()
        train_and_evaluate_classifier(nb_classifier, train_data_pca, test_data_pca, train_labels, test_labels)

        # K-NN Classifier
        print("\nK-NN Classifier:")
        knn_classifier = KNeighborsClassifier()
        train_and_evaluate_classifier(knn_classifier, train_data_pca, test_data_pca, train_labels, test_labels)

        # SVM
        print("\nSVM:")
        svm_classifier = SVC(probability=True)
        train_and_evaluate_classifier(svm_classifier, train_data_pca, test_data_pca, train_labels, test_labels)

if __name__ == "__main__":
    main()
