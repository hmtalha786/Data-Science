#!/usr/bin/env python
# coding: utf-8

# # SciPy Tutorial

# ## Scipy Intoduction
# 
# SciPy ( Scientific Python ) is a collection of mathematical algorithms and convenience functions built on the Numpy extension of Python. It with high-level commands and classes for manipulating and visualizing data. 
# 
# The additional benefit of basing SciPy on Python is that this also makes a powerful programming language available for use in developing sophisticated programs and specialized applications. 
# 
# Let's start with the contents of tutorial.

# ## Contents Covered :
# 
# #### 1) Basic -  Polynomial
# #### 2) Special Funtion 
# #### 3) Integration
# #### 4) Linear Algebra 
# #### 5) Fourier Transformation
# #### 6) Interpolation 
# #### 7) Statistics
# #### 8) Miscellaneous 

# ## 1) Basic - Polynomial

# In[ ]:


from numpy import poly1d
p = poly1d([3,6,8])
print(p)


# In[ ]:


print(p*p)  #(3x^2 + 4^x + 5)^2


# In[ ]:


print(p)
print(" p(2) = ",p(2))
print(" p(2) and p(3) = ",p([2,3]))


# In[ ]:


#Roots of p
p.r


# In[ ]:


#coefficients of p
p 


# In[ ]:


print(p)
p1 = poly1d([1,2,3],variable='y') #equation with variable as y
print()
print(p1)


# In[ ]:


#Integration of a Polynomial

print(p.integ(k=6)) #k = indefinite integration constant


# In[ ]:


# Differentiation of a Polynomial

print(p.deriv())


# ## 2) Special Function (import scipy.special)

# In[ ]:


import scipy.special


# In[ ]:


import numpy as np
a = np.arange(1,11)
a


# In[ ]:


#Cube Root
scipy.special.cbrt(a)


# In[ ]:


# Exponent of 10 (exp10())
x = 3
scipy.special.exp10(x)


# In[ ]:


# Exponent of 2 (exp2())
x = 3
scipy.special.exp2(x)


# In[ ]:


# Degree to Radian (radian(d,m,s))
d = 30
m = 10
s = 50
scipy.special.radian(d,m,s)


# In[ ]:


# Trigonometric Funtions (in degrees)
print("Sine of 30 degrees = ",scipy.special.sindg(30))
print("Cosine of 30 degrees = ",scipy.special.cosdg(30))
print("Tangent of 30 degrees = ",scipy.special.tandg(30))


# In[ ]:


# Permutation
scipy.special.perm(5,2) #5P2 = 20


# In[ ]:


# Combination
scipy.special.comb(5,2) #5C2 = 10


# ## 3) Integration (import scipy.integrate)

# ### Numerous Types of Integration are available in scipy
# 
#    1. quad - Single integration 
#    2. dblquad - Double integration 
#    3. tplquad - Triple integration 
#    4. nquad  - n-fold multiple integration
#    5. and many more...

# In[ ]:


import scipy.integrate


# #### Lambda Function in brief

# In[ ]:


f = lambda x : x**2
f(3)


# In[ ]:


g = lambda x,y : x*y
g(2,3)


# In[ ]:


h = lambda x,y : x**2 + 2*x*y + y**2 #(x+y)^2
h(1,2)  #(1+2)^2


# In[ ]:


# scipy.integrate.quad(func, lower_limit, upper_limit)


# In[ ]:


f = lambda x : x**3
i = scipy.integrate.quad(f, 1, 2)
i


# In[ ]:


from numpy import exp
f= lambda x:exp(-x**3)
i = scipy.integrate.quad(f, 1, 4)
i


# In[ ]:


# Double Integration (Used for area under curve)

#scipy.integrate.dblquad(func, lower_limit, upper_limit, func2, func3)
area = scipy.integrate.dblquad(lambda x, y: x*y, 0, 0.5, lambda x: 0, lambda y: 1-2*y)
area


# ## 4) Linear Algebra (import scipy.linalg)

# In[ ]:


from scipy import linalg


# In[ ]:


import numpy as np


# In[ ]:


a1 = np.array([[1,2],[3,4]])
a1


# In[ ]:


# Determinant of a matrix
linalg.det(a1)


# Compute pivoted LU decomposition of a matrix.
# 
# The decomposition is::
# 
#     A = P L U
# 
# where P is a permutation matrix, L lower triangular with unit
# diagonal elements, and U upper triangular.

# In[ ]:


A = np.array([[1,2,3],[4,5,6],[7,8,8]])
A


# In[ ]:


P, L, U = linalg.lu(A)


# In[ ]:


P


# In[ ]:


L


# In[ ]:


U


# We can find out the eigenvector and eigenvalues of a matrix:

# In[ ]:


EV, EW = linalg.eig(A)


# In[ ]:


#Eigen Values
EV


# In[ ]:


#Eigen Vectors
EW


# Solving systems of linear equations can also be done:

# In[ ]:


B = np.array([[2],[3],[5]])
B


# In[ ]:


A


# In[ ]:


#Solve Linear Equation using Matrix (AX=B)
#linalg.solve()

X = linalg.solve(A,B)


# In[ ]:


X    #Solution : (7/3, 11/3, 1)


# #### Inverse of a matrix

# In[ ]:


A = np.array([[1,3,5],[2,5,1],[2,3,8]])
A


# In[ ]:


#Inverse : linalg.inv()

linalg.inv(A)


# In[ ]:


#Cross Check
# A * inv(A) = I

I = A.dot(linalg.inv(A))
print(np.around(I))


# ## 5) Fourier Transformation (import scipy.fftpack)

# #### What is Fourier Transformation
# The Fourier Transform is a tool that breaks a waveform (a function or signal) into an alternate representation, characterized by sine and cosines. The Fourier Transform shows that any waveform can be re-written as the sum of sinusoidal functions.
# 
# There are two major functions of fftpack :
# - fft : fast fourier transformation
# - ifft : inverse fast fourier transformation

# In[ ]:


import scipy.fftpack


# In[ ]:


import numpy as np
a = np.array([1,5,6,14,25,40])
a


# In[ ]:


# Fourier Transform
y = scipy.fftpack.fft(a)
y


# In[ ]:


# Inverse Fourier Transform
invy = scipy.fftpack.ifft(a)
invy


# In[ ]:


# Cross Verification
i = scipy.fftpack.fft(invy)
print(i)
print()
print(a)


# ## 6) Interpolation (import scipy.interpolate)

# #### What is Interpolation
# interpolation is a type of estimation, a method of constructing new data points within the range of a discrete set of known data points.
# 
# Polynomial interpolation is a method of estimating values between known data points.

# In[ ]:


from scipy.interpolate import interp1d
import numpy as np


# In[ ]:


x = np.arange(0, 6, 1)
y = np.array([0.1, 0.2, 0.3, 0.5, 1.0, 0.9])

xp = np.linspace(0, 5, 100)

import matplotlib.pyplot as plt
plt.plot(x, y, 'bo')

# for linear interpolation
y1 = interp1d(x, y, kind='linear')
plt.plot(xp, y1(xp), 'r-')

# for quadratic interpolation
y2 = interp1d(x, y, kind='quadratic')
plt.plot(xp, y2(xp), 'b--')

# y3 = interp1d(x, y, kind='cubic')
# plt.plot(xp, y3(xp), 'g+')

plt.legend(["Points","Linear","Quadratic"])
plt.show()


# ## 7) Statistics (import scipy.stats)

# ### Common Methods
# 
# The main public methods for continuous RVs are:
# 
# - rvs: Random Variates
# - pdf: Probability Density Function
# - cdf: Cumulative Distribution Function
# - sf: Survival Function (1-CDF)
# - ppf: Percent Point Function (Inverse of CDF)
# - isf: Inverse Survival Function (Inverse of SF)
# - stats: Return mean, variance, (Fisher’s) skew, or (Fisher’s) kurtosis
# - moment: non-central moments of the distribution
# 
# Let’s take a normal RV as an example. 

# ### Probability Distributions

# In[ ]:


from scipy.stats import norm


# In[ ]:


#this gives you a numpy array with 500 elements, 
#randomly valued according to the standard normal distribution (mean = 0, std =1)
x_norm = norm.rvs(size = 500)
print(type(x_norm))


# In[ ]:


norm.mean(), norm.std(), norm.var() #Mean = 0, std = 1


# In[ ]:


#CDF
import numpy as np
norm.cdf(np.array([1,-1., 0, 1, 3, 4, -2, 6])) #Continous Distribution Function


# In[ ]:


#PPF
# It is reverse of CDF
norm.ppf(0.5)   #Percent Point Function


# In[ ]:


x = np.array([1,2,3,4,5,6,7,8,9])
print(x.max(),x.min(),x.mean(),x.var(),sep=",")


# In[ ]:


#Geometric Mean
from scipy.stats.mstats import gmean 
g = gmean([1, 5, 20]) 

print("Geometric Mean is :", g) 


# In[ ]:


#Harmonic Mean
from scipy.stats.mstats import hmean 
h = hmean([1, 5, 20]) 

print("Harmonic Mean is :", h)


# In[ ]:


#Mode 
from scipy import stats
arr1 = np.array([1, 3, 15, 11, 9, 3]) 
print("Arithmetic mode is : ", stats.mode(arr1)) 


# In[ ]:


#Z-Score	
arr1 = [20, 2, 7, 1, 25] 

print ("\nZ-score for arr1 : ", stats.zscore(arr1)) 


# ## 8) Miscellaneous

# #### Scientific Contants

# In[ ]:


scipy.constants.find()


# In[ ]:


#Find Specific Constants
#gives all constants starting with given keyword

scipy.constants.find("Planck")


# In[ ]:


#Physical_Constant method

scipy.constants.physical_constants["Planck constant"]


# In[ ]:


#Returns value of particular Constant

scipy.constants.value("Planck constant")


# In[ ]:


scipy.constants.value("Boltzmann constant")


# In[ ]:


scipy.constants.pi


# ### For more information about SciPy, visit SciPy Official Documentation
# 
# #### https://docs.scipy.org/doc/
