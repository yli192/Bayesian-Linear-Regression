#! /usr/bin/python
import csv
import numpy as np
from scipy.stats import multivariate_normal
import math
#Ye Li
#Machine Learning: Data to Model
#Feb 2016 at JHU
Mat=np.loadtxt(open('stocks.csv','rb'),delimiter=',',skiprows=1)  # Mat[:,0:5] is ms, A, B, C, D, Y
X=Mat[:,1:5]
Y=Mat[:,5:6]
mu_0=np.zeros((4,1)) # 4 x 1
sigma_sq=4
sigma_zero_sq=1
gamma_zero_sq=0.5
I=np.identity(4)
#Activate codes below for model_1##############
Sigma_1=sigma_zero_sq*I
meanVec_1=np.dot(X,mu_0)
covMat_1=np.identity(10000)*sigma_zero_sq+ np.dot(np.dot(X,Sigma_1),X.T)
#N=(1/((2*math.pi)**(10000/2)*np.linalg.norm(covMat_1)**0.5))
posterior_1=multivariate_normal(None,covMat_1)
prob_1=posterior_1.pdf(Y.T[0])
print prob_1

#Activate codes below for model_2##############
Sigma_2=sigma_zero_sq*I
Sigma_2[0,1]=gamma_zero_sq
Sigma_2[1,0]=gamma_zero_sq
meanVec_2=np.dot(X,mu_0)
M=np.dot(X,Sigma_2)
#covMat_2=np.identity(10000)*sigma_zero_sq+ np.dot(M,X.T)
#posterior_2=multivariate_normal(None,covMat_2)
#prob_2=posterior_2.pdf(Y.T[0])
#print prob_2

#Activate codes below for model_3##############
Sigma_3=sigma_zero_sq*I
Sigma_3[2,3]=gamma_zero_sq
Sigma_3[3,2]=gamma_zero_sq
meanVec_3=np.dot(X,mu_0)
#covMat_3=np.identity(10000)*sigma_zero_sq+ np.dot(np.dot(X,Sigma_3),X.T)
#posterior_3=multivariate_normal(None,covMat_2)
#prob_3=posterior_3.pdf(Y.T[0])
#print prob_3

#var = multivariate_normal(None, cov=[[1,0],[0,1]])
#print var
#print var.pdf([1,0])
