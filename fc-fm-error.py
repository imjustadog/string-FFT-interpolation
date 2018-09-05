from math import sin, cos, pi, exp, sqrt, asin, acos, log
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

plt.rc('font',family='Times New Roman',size=10) 

cmap = plt.get_cmap('jet')
#mycolors = ['#ADADAD', '#FFFFFF']
#cmap = mplcolors.LinearSegmentedColormap.from_list(
#        'camp', mycolors, N=2)


fc = range(1,301)
fm = range(1,6001,10)
fs = 250000
N0 = np.zeros((len(fc), len(fm)))
z = np.zeros((len(fc), len(fm)))

for i,fi in enumerate(fc):
    for j,fj in enumerate(fm):
        flag = 0
        temp = (250000 / fi) - int(250000 * 3 / fj) #N needs to <
        temp2 = 250000*30/fj #N needs to <
        temp3 = 250000 / fj #N needs to >
		
        if temp2 < temp:
            temp = temp2
        if temp <= 64:
            temp = 128
            flag = 1
        temp = (int)(log(temp,2))
        if temp > 12:
            temp = 12
            
        #N0[i][j] = temp
        N=int(pow(2,temp))
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
        z[i][j] = log(abs(z[i][j]),10)
        if flag == 1:
            N0[i][j] = 0
        elif z[i][j] >= 2:
            N0[i][j] = 0
        elif N < temp3:
            N0[i][j] = 0
        else:
            N0[i][j] = 1
        if N0[i][j] == 0:
            z[i][j] = 2
        if z[i][j] < -3:
            z[i][j] = -3

figure = plt.figure(figsize=(5,4))

#fm,fc = np.meshgrid(fm,fc)
#figure = plt.figure()
#ax = Axes3D(figure)
#ax.plot_surface(fc, fm, N, rstride=1,cstride=10,cmap = 'rainbow')

ax = plt.gca()
cax = ax.contourf(fm, fc, z,50,
            extend='both',cmap=cmap)
cbar = plt.colorbar(cax)
plt.subplots_adjust(bottom = 0.2,left = 0.13,right=1.0)
plt.xlabel("sensor frequency/Hz")
plt.ylabel("calculate freuency/Hz")
plt.savefig("errorjudge.pdf")
#plt.show()

#for j in range(7,13):
#    print "maximum N is " + str(pow(2,j))
#    for i,f in enumerate(fc):
#        sca = np.where(N[i] >= j)
#        if(len(sca[0]) > 1):
#            print "calculate frequency: " + str(f) + "     sensor frequency:" + str(fm[min(sca[0])]) + '-' + str(fm[max(sca[0])])
#        else:
#            print "calculate frequency: " + str(f) + "     No proper sensor frequency"
#    print "********************************************************************"
