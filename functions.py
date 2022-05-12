"""
Data Analytics 2: Self Study Functions File

@author: Manuel-Milan Ramsler
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import sklearn.linear_model as sk

def dgp1(x_mean,x_std,coefft,coeff,othercoeff,intercept,n):
    """ Data generating process 1
    Input:
        - x: confounding variables
        - coefft: coefficients of confounding variables with treatment
        - coeff: coefficients of confounding variables
        - intercept: intercept
        - othercoeff: coefficients for probability of treatment
        - n: sample size
    Output:
        - betas: outcome, treatment, and confounding variables
    """
    # create covariates
    x = np.random.normal(x_mean,x_std,size = (3,n)).T
    e = np.random.normal(0,1,n)
    # probability of treatment
    treatprob = stats.norm.cdf(x@othercoeff)
    t = np.array(np.random.binomial(1, treatprob,n))
    # results
    y = intercept+t*(coefft)+x@(coeff)+e
    return y, t, x

def dgp2(x_mean,x_std,coefft,coeff,intercept,n,gamma):
    """ Data generating process 2
    Input:
        - x: confounding variables
        - coefft: coefficients of confounding variables with treatment
        - coeff: coefficients of confounding variables
        - intercept: intercept
        - alpha: common support
        - n: sample size
        - gamma: heteroscedasticity
    Output:
        - betas: outcome, treatment, and confounding variables
    """
    # create covariates
    x = np.random.normal(x_mean,x_std,size = (3,n)).T
    e = np.random.normal(0,abs(x@gamma),n)
    # probability of treatment
    treatprob = stats.norm.cdf(x@coeff-2.5)
    t = np.array(np.random.binomial(1, treatprob,n))
    # results
    y = intercept+t*(coefft)+x@(coeff)+e
    return y, t, x

def dgp3(x_mean,x_std,coefft,coeff,intercept,n,othercoeff):
    """ Data generating process 3
    Input:
        - x: confounding variables
        - coefft: coefficients of confounding variables with treatment
        - coeff: coefficients of confounding variables
        - intercept: intercept
        - alpha: common support
        - n: sample size
        - othercoeff: covariates for DGP3
    Output:
        - betas: outcome, treatment, and confounding variables
    """
    # create covariates
    x = np.random.normal(x_mean,x_std,size = (4,n)).T
    e = np.random.normal(0,1,n)
    # probability of treatment
    treatprob = stats.norm.cdf(x@othercoeff)
    t = np.array(np.random.binomial(1, treatprob, n))
    # output
    y = intercept+t*(coefft)+x@(coeff)+e
    x = np.delete(x, 1, axis=1)
    return y, t, x

def myols(x,y):
    """ OLS estimator
    Input:
        - x: covariates
        - y: outcome
    Output:
        - betas: estimated betas
    """
    # combine treatment, covariates and ones
    treatandcovones = np.c_[np.ones(len(y)),x]
    # calculate OLS
    treatandcovonesinv = np.linalg.inv(np.dot(treatandcovones.T, treatandcovones))
    treatandcovonesty = np.dot(treatandcovones.T, y)
    # calculate betas
    betas = np.dot(treatandcovonesinv, treatandcovonesty)
    return(betas)

def mypropensityscore(x,t):
    """ Propensity score
    Input:
        - x: covariates
        - t: treatment
    Output:
        - Propensity score
    """
    # add a column with ones and covariates
    X = np.c_[np.ones(len(x)),x]
    # use package to calculate propensity score
    propensity_scores = sk.LogisticRegression(C=1e6).fit(X, t).predict_proba(X)[:, 1].ravel()
    return(propensity_scores)

def mydoublyrobust(x,t,y):
    """ Doubly robust estimator
    Input:
        - x: covariates
        - t: treatment
        - y: outcome
    Output:
        - Average treatment effect
    """
    # combine outcome, treatment and covariates
    X = np.c_[y,t,x]
    Xnew = np.c_[np.ones(len(x)),x]
    # calculate propensity score
    propensity = mypropensityscore(x, t)
    
    # use OLS for no treatment
    data = pd.DataFrame(X, columns = ["Y", "T", "X1", "X2", "X3"])
    T = data["T"]
    data0 = data[T==0].reset_index(drop=True)
    X0 = data0[["X1", "X2", "X3"]]
    Y0 = data0["Y"]
    betaszero = myols(X0,Y0)
    # estimate the outcome when treatment=0
    treat0 = (Xnew@betaszero)
    
    # use OLS with treatment
    data1 = data[T==1].reset_index(drop=True)
    X1 = data1[["X1", "X2", "X3"]]
    Y1 = data1["Y"]
    betasone = myols(X1,Y1)
    # estimate the outcome when treatment=1
    treat1 = (Xnew@betasone)
    
    # calculate average treatment effect
    step1 = (y-treat1)*t
    step2 = (1-t)*(y-treat0)
    step3 = treat1-treat0+step1/propensity-step2/(1-propensity)
    result_dr = np.mean(step3)
    result_ols = np.mean(treat1-treat0)
    return(result_ols,result_dr)


def mystatisticalresults(results, beta):
    """
    Statistical results
    Input:
        - results: covariates
        - beta: true mean
    Output:
        - Bias and MSE
    """
    # calculate bias and MSE
    bias = np.mean(results) - beta
    mse = np.mean((results-beta)**2)
    print("Bias: "+str(round(bias,4)))
    print("MSE: "+str(round(mse,4)))
    return(bias, mse)

def myplots(results,beta,title):
    # results for OLS and DR
    plt.figure()
    plt.title(title)
    plt.hist(x=results["OLS"], alpha=0.5, color='green',label="OLS")
    plt.hist(x=results["DR"], alpha=0.5, color='blue',label="DR")
    plt.axvline(x=beta,label="True Value", color = "black")
    plt.legend()
    plt.show()
    
