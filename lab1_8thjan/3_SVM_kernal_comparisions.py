import numpy as np
from data import gen_non_linear_data
from visual import plot_2d_data
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC as svm


# KERNEL - this is a trick, which works in such a way that the actaul cordinates of the new dinemsions are not calculated, 
# just the required products are comuted without the mapping 

def main():
    X, y = gen_non_linear_data(n=200)
    plot_2d_data(X, y, title="XOR data")

    linear_svm = svm(kernel='linear')
    linear_svm.fit(X, y)
    linear_y_pred = linear_svm.predict(X)
    linear_accuracy = accuracy_score(y, linear_y_pred)
    print(f"Accuracy linear SVM - XOR: {linear_accuracy}")
    plot_2d_data(X, linear_y_pred, title="XOR Linear SVM prediction")

    poly_svm = svm(kernel='poly', degree=2)
    #maps the data to five dimensions (degree 2) so that it is easier to find patters in the data
    #i.e. it helps find a plain in such a way in the 5D dimnesion
    poly_svm.fit(X,y)
    poly_y_pred = poly_svm.predict(X)
    poly_accuracy = accuracy_score(y, poly_y_pred)
    print(f"Accuracy polynomial SVM - XOR: {poly_accuracy}")
    plot_2d_data(X, poly_y_pred, title="XOR Polynomial SVM prediction")

    rdf_svm = svm(kernel='rbf', gamma='scale')
    #maps data to inifitine dimension space that will help identify complex pattersn in the data
    #it basically fits the guassian distribution on the data
    rdf_svm.fit(X,y)
    rdf_y_pred = rdf_svm.predict(X)
    rdf_accuracy = accuracy_score(y, rdf_y_pred)
    print(f"Accuracy RBF SVM - XOR: {rdf_accuracy}")
    plot_2d_data(X, rdf_y_pred, title="XOR RBF SVM prediction")

    sigmoid_svm = svm(kernel='sigmoid', gamma='scale')
    # uses a tanh activation similar to a shallow neural network
    # introduces non-linearity but is harder to tune and less stable than RBF
    sigmoid_svm.fit(X, y)
    sigmoid_y_pred = sigmoid_svm.predict(X)
    sigmoid_accuracy = accuracy_score(y, sigmoid_y_pred)
    print(f"Accuracy Sigmoid SVM - XOR: {sigmoid_accuracy}")
    plot_2d_data(X, sigmoid_y_pred, title="XOR Sigmoid SVM prediction")

    from sklearn.metrics.pairwise import laplacian_kernel

    K_lap = laplacian_kernel(X, gamma=0.5)

    laplacian_svm = svm(kernel='precomputed')
    # similar to RBF but uses L1 distance instead of L2
    # more robust to outliers and sharp feature differences
    laplacian_svm.fit(K_lap, y)
    laplacian_y_pred = laplacian_svm.predict(K_lap)
    laplacian_accuracy = accuracy_score(y, laplacian_y_pred)
    print(f"Accuracy Laplacian SVM - XOR: {laplacian_accuracy}")
    plot_2d_data(X, laplacian_y_pred, title="XOR Laplacian SVM prediction")

    from sklearn.metrics.pairwise import chi2_kernel

    K_chi2 = chi2_kernel(X, gamma=0.5)

    chi2_svm = svm(kernel='precomputed')
    # measures similarity using chi-square distance
    # works best with non-negative data like histograms or frequency features
    chi2_svm.fit(K_chi2, y)
    chi2_y_pred = chi2_svm.predict(K_chi2)
    chi2_accuracy = accuracy_score(y, chi2_y_pred)
    print(f"Accuracy Chi-Square SVM - XOR: {chi2_accuracy}")
    plot_2d_data(X, chi2_y_pred, title="XOR Chi-Square SVM prediction")

    

if __name__ == "__main__":
    main()
