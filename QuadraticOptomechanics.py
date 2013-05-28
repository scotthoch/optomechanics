import os
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
#This is a first crack as solving for the quadratic optomechanics theory

#**************************************#
# wp domega/dx  #
# wL laser frequency
# m membrane mass
# kappaLR FWHM of left/right cavity modes
# kappaILR coupling rate through mirrors
# Pin incident power
# xbar membrane mean position
# Delta detuning from the resonant frequency assuming no avoided crossing
#
#****************************************************#

def pincL(wp,wL,m,kappaL,kappaIL,Pin):
    return -wp*4*Pin*kappaIL/(wL*m*kappaL**2)

def pincR(wp,wL,m,kappaR,kappaIR,Pin):
    return -wp*4*Pin*kappaIR/(wL*m*kappaR**2)

def alphaBarL(kappaL,kappaR,Delta,wp,xbar,g):
    tempnum=-kappaL*(1j*wp*xbar+1j*Delta-(kappaR/2))
    tempdensqr=2*(g**2+wp**2*xbar**2)
    tempdenomcrs=2*(-1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta+1j*(kappaL/2))*(Delta+1j*(kappaR/2)))
    return tempnum/(tempdensqr+tempdenomcrs)

def alphaBarLconj(kappaL,kappaR,Delta,wp,xbar,g):
    tempnum=-kappaL*(-1j*wp*xbar-1j*Delta-(kappaR/2))
    tempdensqr=2*(g**2+wp**2*xbar**2)
    tempdenomcrs=2*(1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta-1j*(kappaL/2))*(Delta-1j*(kappaR/2)))
    return tempnum/(tempdensqr+tempdenomcrs)

def alphaBarR(kappaL,kappaR,Delta,wp,xbar,g,):
    tempnum=-1j*g*kappaL
    tempdensqr=2*(g**2+wp**2*xbar**2)
    tempdenomcrs=2*(-1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta+1j*(kappaL/2))*(Delta+1j*(kappaR/2)))
    return tempnum/(tempdensqr+tempdenomcrs)

def alphaBarRconj(kappaL,kappaR,Delta,wp,xbar,g,):
    tempnum=1j*g*kappaL
    tempdensqr=2*(g**2+wp**2*xbar**2)
    tempdenomcrs=2*(1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta-1j*(kappaL/2))*(Delta-1j*(kappaR/2)))
    return tempnum/(tempdensqr+tempdenomcrs)


def chiAlphaL(kappaL,kappaR,Delta,wp,xbar,g,omega):
    numnum=wp*((-1j*g**2*kappaL)-(kappaL*(1j*wp*xbar+1j*Delta-(kappaR/2))*(-wp*xbar-Delta-1j*(kappaR/2)+omega)))
    numdenom=2*(g**2+wp**2*xbar**2-1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta+1j*(kappaL/2))*(Delta+1j*(kappaR/2)))
    tnum=numnum/numdenom
    tdenom=g**2-(-wp*xbar+Delta+1j*(kappaL/2)-omega)*(wp*xbar+Delta+1j*(kappaR/2)-omega)
    return tnum/tdenom

def chiAlphaLconj(kappaL,kappaR,Delta,wp,xbar,g,omega):
    numnum=wp*((1j*g**2*kappaL)-(kappaL*(-1j*wp*xbar-1j*Delta-(kappaR/2))*(-wp*xbar-Delta+1j*(kappaR/2)+omega)))
    numdenom=2*(g**2+wp**2*xbar**2+1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta-1j*(kappaL/2))*(Delta-1j*(kappaR/2)))
    tnum=numnum/numdenom
    tdenom=g**2-(-wp*xbar+Delta-1j*(kappaL/2)-omega)*(wp*xbar+Delta-1j*(kappaR/2)-omega)
    return tnum/tdenom

def chiAlphaR(kappaL,kappaR,Delta,wp,xbar,g,omega):
    numnum=wp*(g*kappaL*(1j*wp*xbar+1j*Delta-(kappaR/2))-1j*g*kappaL*(-wp*xbar+Delta+1j*(kappaL/2)-omega))
    numdenom=2*(g**2+wp**2*xbar**2-1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta+1j*(kappaL/2))*(Delta+1j*(kappaR/2)))
    tnum=numnum/numdenom
    tdenom=g**2-(-wp*xbar+Delta+1j*(kappaL/2)-omega)*(wp*xbar+Delta+1j*(kappaR/2)-omega)
    return tnum/tdenom

def chiAlphaRconj(kappaL,kappaR,Delta,wp,xbar,g,omega):
    numnum=wp*(g*kappaL*(-1j*wp*xbar-1j*Delta-(kappaR/2))+1j*g*kappaL*(-wp*xbar+Delta-1j*(kappaL/2)-omega))
    numdenom=2*(g**2+wp**2*xbar**2+1j*wp*xbar*((kappaL/2)-(kappaR/2))-(Delta-1j*(kappaL/2))*(Delta-1j*(kappaR/2)))
    tnum=numnum/numdenom
    tdenom=g**2-(-wp*xbar+Delta-1j*(kappaL/2)-omega)*(wp*xbar+Delta-1j*(kappaR/2)-omega)
    return tnum/tdenom

def Sigma(kappaL,kappaR,Delta,wp,xbar,g,omega,wL,m,kappaIL,kappaIR,Pin):
    sc=.5
    return (-pincL(wp,wL,m,kappaL,kappaIL,Pin)*(alphaBarLconj(kappaL,kappaR,Delta,wp,xbar,g,)*
        chiAlphaL(kappaL,kappaR,Delta,wp,xbar,g,omega)+alphaBarL(kappaL,kappaR,Delta,wp,xbar,g,)*
        chiAlphaLconj(kappaL,kappaR,Delta,wp,xbar,g,-omega))+
        pincR(sc*wp,wL,m,kappaR,kappaIR,Pin)*(alphaBarRconj(kappaL,kappaR,Delta,sc*wp,xbar,g,)*
        chiAlphaR(kappaL,kappaR,Delta,sc*wp,xbar,g,omega)+alphaBarR(kappaL,kappaR,Delta,sc*wp,xbar,g,)*
        chiAlphaRconj(kappaL,kappaR,Delta,sc*wp,xbar,g,-omega)))

########################
##   End definitions  ##
########################



global gammaOpt
global omegaOpt
global DeltaPlot
global pltDet
global xplace
global DeltaUndo


##********************************************************##
##  xplace is a list of all membrane positions displaced  ##
##  from the quadratic point that will be plugged into    ##
##  the code                                              ##
##********************************************************##
xplace=np.linspace(-3e-10, 3e-10, num=1000, endpoint=True, retstep=False)
pltsig=np.zeros((1000,1000),dtype='complex')
DeltaPlot=np.zeros((1000,1000))
pltDet=np.linspace(0*2*pi*100e6,30*2*pi*100e6, num=1000, endpoint=True, retstep=False)


###############################################################################
##    The following 2 lines need to be changed to match system parameters    ##
###############################################################################

omegam=2*pi*30e6 #mechanical frequency
gee=2*pi*150e6 #gap size


for i in range(0,len(xplace)):
    pltsig[:,i]=Sigma(kappaL=2*pi*30e6, kappaR=2*pi*30e6, Delta=pltDet-2*pi*1.5e9, wp=2*pi*3.3e18, xbar=xplace[i], g=gee, omega=omegam, wL=2*pi*2.9979e8/1550e-9, m=4.5e-12, kappaIL=2*pi*30e6, kappaIR=2*pi*30e6, Pin=20e-6)
    DeltaPlot[:,i]=(pltDet-2*pi*1.5e9+2*np.pi*(((1.5e9)**2+(1.7e18*xplace[i])**2)**.5))/(2*np.pi)


##********************************************************##
##  To plot this data by detunings, use DeltaPlot[:,i]    ##
##  with gamma/omegaOpt[i:,i]                             ##
##********************************************************##

omegaOpt=np.real(pltsig)/(2*omegam)
gammaOpt=-np.imag(pltsig)/(omegam)
plt.figure(14)
plt.subplot(2,2,1)
plt.imshow(gammaOpt,interpolation='nearest',extent=[xplace.min(), xplace.max(), pltDet.min()/(2*np.pi), pltDet.max()/(2*np.pi)],
 aspect=2*np.pi*(xplace.max()-xplace.min())/(pltDet.max()-pltDet.min()),
  origin='lower',cmap=plt.cm.seismic)
plt.title('Optomechanical damping')
plt.subplot(2,2,2)
plt.imshow(omegaOpt,interpolation='nearest',extent=[xplace.min(), xplace.max(), pltDet.min()/(2*np.pi), pltDet.max()/(2*np.pi)],
 aspect=2*np.pi*(xplace.max()-xplace.min())/(pltDet.max()-pltDet.min()),
  origin='lower',cmap=plt.cm.seismic)
plt.title('Resonant frequency shift')
plt.subplot(2,2,3)
plt.contourf(xplace, pltDet/(2*np.pi), gammaOpt,cmap=plt.cm.seismic)#gist_rainbow_r)
plt.subplot(2,2,4)
plt.contourf(xplace, pltDet/(2*np.pi), omegaOpt,cmap=plt.cm.seismic)#gist_rainbow_r)



