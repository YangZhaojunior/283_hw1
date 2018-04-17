import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import math
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
import random
import pandas as pd
from numpy import dot
from numpy.linalg import inv

# Create graph
sess = tf.Session()

# =============================================================================
# 1)

sampleNo = 200;

s0 = np.zeros((sampleNo,2))
s1 = np.zeros((sampleNo,2))

mu = np.array([0.,0.])
mean = [0.,0.]
con = [[2.,0.],[0.,1.]]
s0[:,0],s0[:,1] = np.random.multivariate_normal(mean,con,sampleNo).T
#plt.plot(s0[:,0],s0[:,1],'+')
#plt.show()

muA = np.array([-2.,1.])
muB = np.array([3.,2.])

s1_1 = np.zeros((sampleNo,2))
s1_2 = np.zeros((sampleNo,2))

mean_cla = [-2.,1.]
mean_clb = [3.,2.]
con_cla = [[1.125,0.875],[0.875,1.125]]
con_clb = [[2.,1.],[1.,2.]]
s1_1[:,0],s1_1[:,1] = np.random.multivariate_normal(mean_cla,con_cla,sampleNo).T
s1_2[:,0],s1_2[:,1] = np.random.multivariate_normal(mean_clb,con_clb,sampleNo).T

for i in range(sampleNo):
    a = random.randint(0,8)
    if a<3:
        s1[i,0] = s1_1[i,0]
        s1[i,1] = s1_1[i,1]
    else:
        s1[i,0] = s1_2[i,0]
        s1[i,1] = s1_2[i,1]

#plt.plot(s0[:,0],s0[:,1],'g+')
#plt.plot(s1[:,0],s1[:,1],'rx')
#plt.show()

total_xs = np.concatenate((s0,s1))
tmp0 = np.zeros((sampleNo,1))
tmp1 = np.ones((sampleNo,1))
total_ys = np.concatenate((tmp0,tmp1))

# =============================================================================
# 7)

total_xs_nk = 1.0*np.ones((sampleNo*2,10))
total_xs_nk[:,1]=total_xs[:,0]
total_xs_nk[:,2]=total_xs[:,1]
total_xs_nk[:,3]=total_xs[:,0]*total_xs[:,0]
total_xs_nk[:,4]=total_xs[:,1]*total_xs[:,1]
total_xs_nk[:,5]=total_xs[:,0]*total_xs[:,1]
total_xs_nk[:,6]=total_xs[:,0]*total_xs[:,0]*total_xs[:,0]
total_xs_nk[:,7]=total_xs[:,0]*total_xs[:,0]*total_xs[:,1]
total_xs_nk[:,8]=total_xs[:,0]*total_xs[:,1]*total_xs[:,1]
total_xs_nk[:,9]=total_xs[:,1]*total_xs[:,1]*total_xs[:,1]

def logit(x):
    return 1.0/(1+np.exp(-1.0*x))

X = 1.0*np.zeros((sampleNo*2,10))
Y = 1.0*np.zeros((sampleNo*2,1))
h = 1.0*np.zeros((sampleNo*2,1))
X = total_xs_nk
Y[:,0] = total_ys[:,0]

m,n = X.shape
alpha = 0.0065
theta_g = 1.0*np.zeros((n,1))
maxCycles = 3000
J = pd.Series(np.arange(maxCycles, dtype = float))
lambda2 = 1

# Gradient
for i in range(maxCycles):
    h = logit(dot(X, theta_g)) # estimated     
    error = h - Y # error
    grad = dot(X.T, error) + lambda2*theta_g # gradient
    theta_g -= alpha * grad
#print (theta_g)

tmp10 = 1.0*np.zeros((10,1))
tmp10[0,0] = 1.0

# Newton
theta_n = 1.0*np.zeros((n,1)) #initial parameters
maxCycles = 1 # iteration # 
for i in range(maxCycles):
    h = logit(dot(X, theta_n)) # estimated  
    error = h - Y # error
    grad = dot(X.T, error) + lambda2*theta_n # gradient
    A =  h*(1-h)* np.eye(len(X)) 
    H = np.mat(X.T)* A * np.mat(X) + lambda2*(np.eye(10)-tmp10) # Hessian, H = X`AX
    theta_n -= inv(H)*grad
#print (theta_n)

# =============================================================================
# 8)

h = 0.02
x0_min,x0_max = total_xs[:,0].min()-1, total_xs[:,0].max()+1
x1_min,x1_max = total_xs[:,1].min()-1, total_xs[:,1].max()+1
xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, h),
                     np.arange(x1_min, x1_max, h))
i_range,j_range = xx0.shape
zz = np.zeros((i_range,j_range))
for i in range(i_range):
    for j in range(j_range):
        x0_now = xx0[i,j]
        x1_now = xx1[i,j]
        zz[i,j] = dot(theta_g.T,[1,x0_now,x1_now,x0_now*x0_now,x1_now*x1_now,x0_now*x1_now,x0_now*x0_now*x0_now,x0_now*x0_now*x1_now,x0_now*x1_now*x1_now,x1_now*x1_now*x1_now])
#print (theta_n)
plt.contour(xx0,xx1,zz,[0],cmap=plt.cm.Paired)
plt.plot(s0[:,0],s0[:,1],'gx')
plt.plot(s1[:,0],s1[:,1],'r+')
plt.show()
 
# =============================================================================
# 9)

#j = 0
#for i in range(sampleNo):
#    x0_now = total_xs[i,0]
#    x1_now = total_xs[i,1]
#    if dot(theta_n.T,[1.0,x0_now,x1_now,x0_now*x0_now,x1_now*x1_now,x0_now*x1_now,x0_now*x0_now*x0_now,x0_now*x0_now*x1_now,x0_now*x1_now*x1_now,x1_now*x1_now*x1_now]) >=0:
#        j +=1
#for i in range(sampleNo):
#    x0_now = total_xs[i+sampleNo,0]
#    x1_now = total_xs[i+sampleNo,1]
#    if dot(theta_n.T,[1.0,x0_now,x1_now,x0_now*x0_now,x1_now*x1_now,x0_now*x1_now,x0_now*x0_now*x0_now,x0_now*x0_now*x1_now,x0_now*x1_now*x1_now,x1_now*x1_now*x1_now]) <0:
#        j +=1
#
#error_ratio = (1.0*j)/(2.0*sampleNo)
#print (error_ratio)

## ------------------------

total_xs_no1 = np.load("total_xs_no1.npy")
total_ys_no1 = np.load("total_ys_no1.npy")

i,j = total_xs_no1.shape
sampleNo = int(i/2)
j = 0
for i in range(sampleNo):
    x0_now = total_xs_no1[i,0]
    x1_now = total_xs_no1[i,1]
    if dot(theta_n.T,[1,x0_now,x1_now,x0_now*x0_now,x1_now*x1_now,x0_now*x1_now,x0_now*x0_now*x0_now,x0_now*x0_now*x1_now,x0_now*x1_now*x1_now,x1_now*x1_now*x1_now]) >=0:
        j +=1
for i in range(sampleNo):
    x0_now = total_xs_no1[i+sampleNo,0]
    x1_now = total_xs_no1[i+sampleNo,1]
    if dot(theta_n.T,[1,x0_now,x1_now,x0_now*x0_now,x1_now*x1_now,x0_now*x1_now,x0_now*x0_now*x0_now,x0_now*x0_now*x1_now,x0_now*x1_now*x1_now,x1_now*x1_now*x1_now]) <0:
        j +=1

error_ratio = (1.0*j)/(2.0*sampleNo)
print (error_ratio)