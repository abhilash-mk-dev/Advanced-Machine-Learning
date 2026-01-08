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

if __name__ == "__main__":
    main()
