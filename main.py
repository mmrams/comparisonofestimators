"""
Data Analytics 2: Self Study Main File

@author: Manuel-Milan Ramsler
"""
# import necessary packages
from numpy.random import seed
import pandas as pd

# load functions file
import os
os.chdir('E:\\Switch Drive Files\\Institution\\Semester 2\\Data Analytics 2\\Personal Project\\')
import functions as fc

# seed to ensure program provides consistent results
seed(1234) 
# parameters of model
# mean
x_mean = 1
# standard deviations
x_std =1
# coefficients of confounding variables with treatment
coefft = 1
# coefficients of confounding variables
coeff =[0.5,0.5,0.5]
# coefficients for probability of treatment
othercoeff = [1,1,1]
# coefficients for data generating process 3
coeffdgp3 = [1,1,0.1,0.1]
# coefficients for probability of treatment for data generating process 3
othercoeffdgp3 = [1,1,0.1, 0.1]
# heteroscedasticity for data generating process 3
gamma=[0.3,0.3,0.7]
# intercept
intercept = 5
# sample size
n = 500
# number of simulations
num_simulations = 100

# conduct data generating processes
datadgp1 = pd.DataFrame(columns = ["OLS", "DR"], index = range(num_simulations))
datadgp2 = pd.DataFrame(columns = ["OLS", "DR"], index = range(num_simulations))
datadgp3 = pd.DataFrame(columns = ["OLS", "DR"], index = range(num_simulations))
# conduct for number of simulations
for i in range(num_simulations):
    Y1,T1,X1 = fc.dgp1(x_mean,x_std,coefft,coeff, othercoeff,intercept,n)
    datadgp1.iloc[i] = fc.mydoublyrobust(X1,T1,Y1)
    Y2,T2,X2 = fc.dgp2(x_mean,x_std,coefft,coeff,intercept,n,gamma)
    datadgp2.iloc[i] = fc.mydoublyrobust(X2,T2,Y2)
    Y3,T3,X3 = fc.dgp3(x_mean,x_std,coefft,coeffdgp3,intercept,n,othercoeffdgp3)
    datadgp3.iloc[i] = fc.mydoublyrobust(X3,T3,Y3)

# print statistical results
print("DGP1")
print("OLS")
fc.mystatisticalresults(datadgp1["OLS"],1)
print("DR")
fc.mystatisticalresults(datadgp1["DR"],1)
print("DGP2")
print("OLS")
fc.mystatisticalresults(datadgp2["OLS"],1)
print("DR")
fc.mystatisticalresults(datadgp2["DR"],1)
print("DGP3")
print("OLS")
fc.mystatisticalresults(datadgp3["OLS"],1)
print("DR")
fc.mystatisticalresults(datadgp3["DR"],1)

# plot results
fc.myplots(datadgp1,1,"DGP1")
fc.myplots(datadgp2,1,"DGP2")
fc.myplots(datadgp3,1,"DGP3")
