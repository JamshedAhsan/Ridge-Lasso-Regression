# let's import all necessary libs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from sklearn.linera_model import LinearRegresion, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

import warnings 
warnings.filterwarnings('ignore')

# Let's create sample data for predicting sales given the spending on marketing

data = {'Marketing Spend (Millin $)' : [23, 26, 30, 34, 43, 48],
       'Sales (Million $)' : [651, 762, 856, 1063, 1190, 1298]}

data = pd.DataFrame(data)
data

#plotting scatter plot to visualize the data
sns.scatterplot(data, x='Marketing Spend (Millin $)', y='Sales (Million $)')

#Scaling the data between 0 and 1 using MinMaxScaler
scaler = MinMaxScaler()

data[['Marketing Spend (Millin $)', 'Sales (Million $)']] = scaler.fit_transform(data[['Marketing Spend (Millin $)', 'Sales (Million $)']])
data


#plotting scatter plot to visualize the data
sns.scatterplot(data, x='Marketing Spend (Millin $)', y='Sales (Million $)')

# Let's build the simple linear regression model
X = data['Marketing Spend (Millin $)'].values.reshape(-1, 1)
y = data[]
y = data['Sales (Million $)']

reg = LinearRegression()
reg.fit(X,y)

#predict using the model
y_pred = reg.predict(X)
y_pred

# Let's check the model performance
r2_score(y, y_pred)

rss = np.sum(np.square(y - y_pred))
print(rss)
mse = mean_squared_error(y, y_pred)
print(mse)
rmse = mse**0.5

#let's plot Predicted Sales vs Marketing Spend

plt.scatter(X, y, color = 'blue') #origina
plt.scatter(X, y_pred, color = 'red, linewidth=3) #prdiction by model
plt.xlabel('Marketing Spend (Millin $)')
plt.ylabel('Predicted Sales (Millinon $')
plt.show()
            
#now let's try to fit a  polynomial regression model on this model

# 3 specifies the degree of polynomail to be generated
poly = PolynomialFeatures(3) 
# Transform the variable X to 1, X, X^2, X^3
X_poly3 = poly.fit_transform(X)  
            
poly_degree = 5 #arrived at degree 5 with trial & error; let' use it for learning purpose for now
polyreg5 = PolynomialFeatures(poly_degree)
X_poly5 = polyreg5.fit_transform(X)
linreg5 = LinearRegression()
linreg5.fit(X_poly5, y)
            
#let's plot the plolynomial regression and simple linear regression
#300 represents equal spaced 300 valuees between 0 and 1
X_seq = np.linspace(X.min(), X.max(), 300).reshape(-1, 1) 
plt.figure()
plt.scatter(X, y)
#let' fit the polynomial reg model
plt.plot(X_seq, linreg5.predict(polyreg5.fit_transform(X_seq)), color='black')
#let's fit the regression model
plt.plot(X_seq, reg.predict(X_seq), color='red)          
plt.title("Polynomial Regression with degree" +str(degree))
plt.xlabel("Marketing Spend (Million $)")
plt.ylable("Sales (Million $)")            
plt.show()            

y_pred5 = linreg5.predict(polyreg5.fit_transfor(X))         
print(r2_score(y, y_pred5)         
            
rss = np.sum(np.square(y - y_pred5))
print(rss)
mse = mean_squared_error(y, y_pred5)
print(mse)
rmse = mse**0.5      
            
#we created an intetional overfit model with r2_score value of 1
# since we know overfit model shall have high variance for unseen data
      
# step 3: let's apply ridge regularization

# let's apply ridge regularization with varying hyperparameter "lambda"

#let's make list of possible lambda for trial 
      
lamdas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]     
      
#note: each lambda shall give us set of co-eff
for i in lambdas:
      degree = 5
      ridgecoeff = PolynomialFeatures(degree)
      #transform X to polynomail features
      X_poly = ridgecoeff.fit_transform(X)
      #initialize Ridge Regression with specific value of lambda
      ridgereg = Ridge(alpha=i)
      ridgereg.fit(X_ply, y)
      
      #let's plot
      plt.figure()
      plt.scatter(X, y)
      plt.plot(X_seq, ridgereg.predict(ridgecoff.fit_transform(X_seq)), color='black')
      plt.plot(X_seq, reg.predict(X_seq), color='red')
      plt.title("Polynomial Regression with degress" + str(degress)) + "and lambda=" +str(i))
      plt.show()
            
      #compute the r2 score
      y_pred = ridgereg.predict(X_poly)
      print("r2 score= " + str(r2_score(y, y_pred)))
      print(ridgecoff.coef)
            
# step 4: let's apply Lasso regularization

# let's apply ridge regularization with varying hyperparameter "lambda"

#let's make list of possible lambda for trial 
      
lamdas = [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]     
      
#note: each lambda shall give us set of co-eff
for i in lambdas:
      degree = 5
      lassocoeff = PolynomialFeatures(degree)
      #transform X to polynomail features
      X_poly = lassocoeff.fit_transform(X)
      #initialize Ridge Regression with specific value of lambda
      lassoreg = Lasso(alpha=i)
      lassoreg.fit(X_ply, y)
      
      #let's plot
      plt.figure()
      plt.scatter(X, y)
      plt.plot(X_seq, lassoreg.predict(lassocoeff.fit_transform(X_seq)), color='black')
      plt.plot(X_seq, reg.predict(X_seq), color='red')
      plt.title("Polynomial Regression with degress" + str(degress)) + "and lambda=" +str(i))
      plt.show()
            
      #compute the r2 score
      y_pred = lassoreg.predict(X_poly)
      print("r2 score= " + str(r2_score(y, y_pred)))
      print(ridgecoff.coef)
