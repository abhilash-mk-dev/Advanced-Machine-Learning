import matplotlib.pyplot as plt
import numpy as np

def plot_2d_data(X,y, title="Linear Data"):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

def plot_margin(X, y, svm, title = "Margin Plot"):
    plt.figure(figsize=(7,6))
    plt.scatter(X[:,0], X[:,1], c=y, cmap="bwr", edgecolors="k")

    plt.scatter(
        svm.support_vectors_[:,0],
        svm.support_vectors_[:,1],
        s=120, facecolors = "none", edgecolors="k", linewidths=2,
        label = "Support vector"
    )

    w=svm.coef_[0]
    b=svm.intercept_[0]

    x_vals = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)

    y_decision = -(w[0]*x_vals+b)/w[1]

    y_margin_pos = -(w[0]*x_vals+b - 1)/w[1]
    y_margin_neg = -(w[0]*x_vals+b + 1)/w[1]

    plt.plot(x_vals, y_decision, "k-", label = "Decision Boundry")
    plt.plot(x_vals, y_margin_pos, "k--", label = "Margin +1y")
    plt.plot(x_vals, y_margin_neg, "k--", label = "Margin -1")

    plt.fill_between(
        x_vals, y_margin_pos, y_margin_neg,
        color = "gray", alpha = 0.2, label = "Margine Area"
    )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Hard margie SVM with Margin Visualizatiob")
    plt.legend()
    plt.show()