from math import sin, cos, pi, exp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

plt.rc('font',family='Times New Roman',size=10) 

fs = 250000
x = range(6,13)
fm = range(500,6000,2)
z = [[0 for i in range(len(fm))] for j in range(len(x))]

for i,n in enumerate(x):
    for j,fj in enumerate(fm):
        N=pow(2,n)
        w = np.hanning(N)
        ts=np.array(range(0,N))/float(fs)
        s=[cos(2*pi*fj*t) for t in ts]
        s=np.multiply(s,w)
        xf=np.fft.fft(s)
        xf=np.abs(xf[0:int(N/2)])/(N/2)
        xf[0] = xf[0] / 2
        max_index = np.where(xf==max(xf))
        max_index = max_index[0][0]
        if xf[max_index-1] > xf[max_index+1]:
            temp = (xf[max_index]/xf[max_index-1]-2)/(1+xf[max_index]/xf[max_index-1]);
            z[i][j] = (max_index + temp)*fs/N-fj
        elif xf[max_index-1] < xf[max_index+1]:
            temp = (xf[max_index]/xf[max_index+1]-2)/(1+xf[max_index]/xf[max_index+1]);
            z[i][j] = (max_index - temp)*fs/N-fj
        else:
            z[i][j] = max_index*fs/N - fj
        z[i][j] = np.abs(z[i][j])

d = [z[i][:] for i in range(len(z))]
max_pol = [[] for j in range(len(x))]
min_pol = [[] for j in range(len(x))]
for i in range(len(x)):
    for j in range(len(fm) - 2):
        if z[i][j] < z[i][j + 1] and z[i][j + 2] < z[i][j + 1]:
            max_pol[i].append((j + 1,z[i][j + 1]))
        if z[i][j] > z[i][j + 1] and z[i][j + 2] > z[i][j + 1]:
            min_pol[i].append((j + 1,z[i][j + 1]))

for i in range(len(x)):
    min_pol[i].append((len(fm) - 1,z[i][len(fm) - 1]))
    max_pol[i].append((max_pol[i][-1][0],max_pol[i][-1][1]))
    if min_pol[i][0][0] > max_pol[i][0][0]:
        del max_pol[i][0]

for i in range(len(x)):
    d[i][-1] = max_pol[i][-1][1]
    for j in range(len(min_pol[i]) - 1):
        for k in range(min_pol[i][j][0],min_pol[i][j + 1][0]):
            d[i][k] = max_pol[i][j][1]


for i,x in enumerate(x):
    fig = plt.figure(figsize=(4,3))
    plt.plot(fm,z[i],'-')
    plt.plot(fm,d[i],'-')
    plt.subplots_adjust(bottom = 0.2,left = 0.15,right=0.90)
    plt.ylabel("error/Hz")
    plt.xlabel("frequency/Hz")
    plt.show()
