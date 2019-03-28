import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from sympy import plot_implicit
from mpl_toolkits.mplot3d import Axes3D



# fig = plt.figure()
# ax = Axes3D(fig)

mean1 = [1, 1]
cov1 = [[6,3], [3, 6]]
data1 = np.random.multivariate_normal(mean1, cov1, 15)
print(data1)
SI1 = np.linalg.inv(cov1)
d1 = []
n = data1.shape[0]
mean2 = [10, 10]
cov2 = [[5, 2], [2, 5]]
data2 = np.random.multivariate_normal(mean2, cov2, 15)
print(data2)
SI2 = np.linalg.inv(cov2)

d2 = []
x=np.arange(-50,50,1)
y=np.arange(-50,50,1)
X, Y = np.meshgrid(x, y)
# ezplot = lambda exper: plot_implicit(parse_expr(exper)) # 用了匿名函数
Z = np.dot(np.square(X-mean2[0]),SI2[0,0]) + np.dot(np.dot((Y-mean2[1]),(X-mean2[0])),SI2[1,0])*2 + np.dot(np.square(Y-mean2[1]),SI2[1,1]) - (np.dot(np.square(X-mean1[0]),SI1[0,0]) + np.dot(np.dot((Y-mean1[1]),(X-mean1[0])),SI1[1,0])*2 + np.dot(np.square(Y-mean1[1]),SI1[1,1]))
# Z = '((x-mean2[0])**2)*SI2[0,0]+(y-mean2[1])*(x-mean2[0])*SI2[1,0]*2+((y-mean2[1])**2)*SI2[1,1]-(((x-mean1[0])**2)*SI1[0,0]+(y-mean1[1])*(x-mean1[0])*SI1[1,0]*2+((y-mean1[1])**2)*SI1[1,1])'
# ezplot(Z)
for i in range(0,n):
    delta11 = data1[i] - mean1
    delta12 = data1[i] - mean2
    delta21 = data2[i] - mean1
    delta22 = data2[i] - mean2
    w1 = np.dot(np.dot(delta12,SI2),delta12.T) - np.dot(np.dot(delta11,SI1),delta11.T)
    w2 = np.dot(np.dot(delta22,SI2),delta22.T) - np.dot(np.dot(delta21,SI1),delta21.T)
    d1.append(w1)
    d2.append(w2)
print(d1)
print(d2)

# plt.xlabel('x')
# plt.ylabel('y')
plt.xlim(xmax=max(np.hstack((data1[:,0],data2[:,0]))),xmin=min(np.hstack((data1[:,0],data2[:,0]))))
plt.ylim(ymax=max(np.hstack((data1[:,1],data2[:,1]))),ymin=min(np.hstack((data1[:,1],data2[:,1]))))
# plt.plot(np.hstack((data1[:,0],data2[:,0])),np.hstack((data1[:,1],data2[:,1])),'ro')
plt.plot(data1[:,0],data1[:,1],'ro',color='green')
plt.plot(data2[:,0],data2[:,1],'ro',color='red')
plt.contour(X,Y,Z,0)
plt.show()



