# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:19:05 2018

@author: wanly
"""
import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
def Xkgenerator(Xklast,a,Sigma,DeltaT):
    """
    generate Xk
    float Xklast:xk-1
    float a : a
    float DeltaT
    """
    mu=np.exp(-1*a*DeltaT)*Xklast
    sigma=np.sqrt((np.power(Sigma,2)/(2*a))*(1-np.exp(-2*a*DeltaT)))
    ret=np.random.normal(mu,sigma)
    return ret

def Xgenerator(T0,T,DeltaT,a,Sigma):
    steps=int((T-T0)/DeltaT)
    x=[]
    for i in range(steps):
        if i==0:
            x.append(0)
            continue
        x.append(Xkgenerator(x[i-1],a,Sigma,DeltaT))
    return x

# =============================================================================
# 
# def GenerateW(T0,T,DeltaT):
#     """
#     generate wiener process
#     float T0,T,DeltaT
#     """
#     steps=int((T-T0)/DeltaT)
#     
#     W=[] #steps from W[t]:ti->ti+1
#     for i in range(steps):
#         if i == 0:
#             W.append(np.random.normal()*np.sqrt(DeltaT))
#             continue
#         W.append(W[i-1]+np.random.normal()*np.sqrt(DeltaT))
#     return W
# =============================================================================

def GenerateSteps(T0,T,DeltaT,a):
    Steps=[]
    W=[]
    steps=int((T-T0)/DeltaT)
    for i in range(steps):
        if i==0:
            temp=(1/(2*a))*(1-np.exp(-2*a*DeltaT))
            tempc=(1/a)*(1-np.exp(-1*a*DeltaT))
            mean=[0,0]
            cov=[[temp,tempc],[tempc,DeltaT]]
            x,y=np.random.multivariate_normal(mean,cov)     
            Steps.append(x)
            W.append(y)
            continue
        temp=(1/(2*a))*(1-np.exp(-2*a*DeltaT))
        tempc=(1/a)*(1-np.exp(-1*a*DeltaT))
        mean=[0,0]
        cov=[[temp,tempc],[tempc,DeltaT]]
        x,y=np.random.multivariate_normal(mean,cov) 
        W.append(W[i-1]+y)
        Steps.append(x)
    return Steps,W

def V(t1,t2,Sigma,a):
    ret=np.power(Sigma,2)/np.power(a,2)
    ret=ret*(t2-t1-3/(2*a)+(2/a)*np.exp(-1*a*(t2-t1))-(1/(2*a))*np.exp(-2*a*(t2-t1)))
    return ret

#p(tk-1,tk)=exp(part1)*part2*exp(part3)

def part1(x,W,Steps,a,DeltaT,Sigma,T0,T):
    steps=int((T-T0)/DeltaT)
    part1=[]
    for i in range(steps):
        if i==0:
            temp=(1-np.exp(-1*a*DeltaT))/a
            temp=temp*x[i]
            temp=temp+(Sigma/a)*(W[i])
            temp=temp-(Sigma/a)*Steps[i]
            part1.append(temp)
            continue
        temp=(1-np.exp(-1*a*DeltaT))/a
        temp=temp*x[i]
        temp=temp+(Sigma/a)*(W[i]-W[i-1])
        temp=temp-(Sigma/a)*Steps[i]
        part1.append(temp)
    return part1

def part2(p):
    #list discount factor p 0~T N+1!
    part2=[]
    for i in range(len(p)-1):
        part2.append(p[i+1]/p[i])
    return part2

def part3(T0,T,DeltaT):
    steps=int((T-T0)/DeltaT)
    part3=[]
    for i in range(steps):
        temp=-(1/2)*(V(T0,T0+(i+1)*DeltaT,Sigma,a)-V(T0,T0+(i)*DeltaT,Sigma,a))
        part3.append(temp)
    return part3

def InstantDicount(T0,T,DeltaT,Sigma,a,p):
    #p(t1,t2)
    x=Xgenerator(T0,T,DeltaT,a,Sigma)
    Steps,W=GenerateSteps(T0,T,DeltaT,a)
    parti1=part1(x,W,Steps,a,DeltaT,Sigma,T0,T)
    parti2=part2(p)
    parti3=part3(T0,T,DeltaT)
    steps=int((T-T0)/DeltaT)
    ret=[]
    for i in range(steps):
        temp=np.exp(-1*parti1[i])
        temp=temp*parti2[i]
        temp=temp*np.exp(parti3[i])
        ret.append(temp)
    return ret

def DDgenerator(InstantDiscount0):
    ret=[]
    for i in range(len(InstantDicount0)):
        if i==0:
            ret.append(InstantDicount0[i])
            continue
        ret.append(ret[i-1]*InstantDicount0[i])
    return ret

def Dgenerator(DD):
    ret=[1/x for x in DD]
    return ret

def MyEquation(ps,k,InstantR0,tindex):
    return ps[tindex+1]*(InstantR0[tindex]-k)

def StandardPricerhelper(T0,T,DeltaT,Sigma,a,p,n,k,t):
    summer=0
    tindex=int(t/DeltaT-1)
    for i in range(n):
        InstantDicount0=InstantDicount(T0,T,DeltaT,Sigma,a,p)
        DD=DDgenerator(InstantDicount0)
        InstantR0=InstantR(DD,DeltaT)
        ps=DD
        summer=summer+MyEquation(ps,k,InstantR0,tindex)
    return summer
    
def StandardPricer(T0,T,DeltaT,Sigma,a,p,n,t):
    FRArate=fsolve(lambda k:StandardPricerhelper(T0,T,DeltaT,Sigma,a,p,n,k,t),[0.05])
    return FRArate

def MyEquation2(ps,k,InstantR0,tindex):
    return ps[tindex]*(InstantR0[tindex]-k)

def StandardPricerhelper2(T0,T,DeltaT,Sigma,a,p,n,k,t):
    summer=0
    tindex=int(t/DeltaT-1)
    for i in range(n):
        InstantDicount0=InstantDicount(T0,T,DeltaT,Sigma,a,p)
        DD=DDgenerator(InstantDicount0)
        InstantR0=InstantR(DD,DeltaT)
        ps=DD
        summer=summer+MyEquation2(ps,k,InstantR0,tindex)
    return summer
    
def StandardPricer2(T0,T,DeltaT,Sigma,a,p,n,t):
    FRArate=fsolve(lambda k:StandardPricerhelper2(T0,T,DeltaT,Sigma,a,p,n,k,t),[0.01])
    return FRArate

def InstantR(DD,DeltaT):
    InstantR=[]
    for i in range(len(DD)):
        if i==0:
            InstantR.append((1-DD[i])/(DeltaT*DD[i]))
            continue
        InstantR.append((DD[i-1]-DD[i])/(DeltaT*DD[i]))
    return InstantR

def pp(T0,T,DeltaT,Sigma,a,p,n,t):
    tindex=int(t/DeltaT)
    rs=[]
    rr=[]
    for i in range(n):
        InstantDicount0=InstantDicount(T0,T,DeltaT,Sigma,a,p)
        DD=DDgenerator(InstantDicount0)
        InstantR0=InstantR(DD,DeltaT)
        rs.append(InstantR0[tindex])
        rr.append(DD[-1])
    return np.mean(rr)
    
if __name__ == '__main__':
    T0=0
    T=10
    DeltaT=0.25
    a=1
    Sigma=0.02
    n=int((T-T0)/DeltaT)
    p=[1-i*0.5/(n+1) for i in range(n+1)]
    InstantDicount0=InstantDicount(T0,T,DeltaT,Sigma,a,p)
    DD=DDgenerator(InstantDicount0)
    InstantR0=InstantR(DD,DeltaT)

        
    nn=1000
    print(DD)
    t=1
# =============================================================================
#     print(StandardPricer(T0,T,DeltaT,Sigma,a,p,nn,t))
#     print(StandardPricer2(T0,T,DeltaT,Sigma,a,p,nn,t))
# =============================================================================
    print(pp(T0,T,DeltaT,Sigma,a,p,nn,t))
    
    print((p[4]-p[5])/(0.25*p[5]))
    plt.plot(DD)
    plt.show()
    plt.plot(p)
    #print(Dgenerator(DD))
    #standardFRAprice=