import numpy as np
from data import gen_non_linear_data
from visual import plot_2d_data
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

X,y=gen_non_linear_data(n=200)
plot_2d_data(X,y,title="XOR data")

#Here we are traning a linear model on non linear data
#dispite this is againts general inuation, we are doiong this to demonstrate accuracy variation 
#before and after applying trasformations to reduce the dimensions

model = LogisticRegression()
model.fit(X,y)
linear_y_pred = model.predict(X)
Linear_accuracy = accuracy_score(y,linear_y_pred)

#accuracy of linear data without any transformation
print(f"Accuracy linear - XOR :", Linear_accuracy)
plot_2d_data(X,linear_y_pred, title = "XOR Linear model prediction")

poly = PolynomialFeatures(degree=2, include_bias=False)
#degree 2 is maximum degree or power to which diffrent fearures are generated
#exmaple x1, x2 --> x1,x2,x1^2, x2^2, x1*x2
#number of features is given by (n+d d) - 1
#we can also include a bias feature which will have same value no matter what the x1 and x2 values are
X_poly = poly.fit_transform(X)

p_model = LogisticRegression()
p_model.fit(X_poly,y)
p_linear_y_pred = p_model.predict(X_poly)
p_Linear_accuracy = accuracy_score(y,p_linear_y_pred)

#accuracy of linear data with transformation
print(f"Accuracy linear - XOR :", p_Linear_accuracy)
plot_2d_data(X,p_linear_y_pred, title = "XOR Linear model prediction")