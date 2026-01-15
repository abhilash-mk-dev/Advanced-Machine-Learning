from sklearn.svm import SVC
from data import gen_linear_data
from visual import plot_margin
from visual import plot_margin

X , y = gen_linear_data(100, 1.0)
svm_hard = SVC(kernel = "linear", C = 1e6)
svm_hard.fit(X, y)

print("Number of support vecotrs Hard Margin (std = 1):", len(svm_hard.support_vectors_))
plot_margin(X, y, svm_hard, title="Hard margin plot for Non-overlapping data")


X_overlap , y_overlap = gen_linear_data(100, 3.0)

svm_hard = SVC(kernel = "linear", C = 1e6)
svm_hard.fit(X_overlap, y_overlap)

print("Number of support vecotrs:", len(svm_hard.support_vectors_))
plot_margin(X_overlap, y_overlap, svm_hard, title="Hard margin plot for overlapping data")

svm_soft = SVC(kernel = "linear", C = 1.0) #try for diffrent values
svm_soft.fit(X_overlap, y_overlap)

print("Number of support vecotrs:", len(svm_soft.support_vectors_))
plot_margin(X_overlap, y_overlap, svm_soft, title="Soft margin plot for overlapping data")