from django.shortcuts import render,redirect
import math
from django.http import JsonResponse 
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scipy.stats import norm
import numpy as np    

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)
 
# class BinomialModel(object): 
#     def __init__(self, s0, u, d, strike, maturity, rfr,  n, compd = "s", dyield = None):
#         self.s0 = s0
#         self.u = u
#         self.d = d
#         self.rfr = rfr
#         self.maturity = maturity 
#         self.strike = strike
#         self.n = n
#         self.compd = compd
#         self.dyield = dyield
    
#     def call_price(self):
#         delta = float(self.maturity)/float(self.n)
        
#         if self.compd == "c":
#             if self.dyield ==None: 
#                 q = (math.exp(self.rfr*delta) - self.d) / (self.u - self.d)
#             else:
#                 q = (math.exp((self.rfr-self.dyield)*delta) - self.d) / (self.u - self.d)
#         if self.compd == "s":
#             if self.dyield == None: 
#                 q = (1 + self.rfr*delta - self.d) / (self.u - self.d)
#             else:
#                 q = (1+ (self.rfr - self.dyield)*delta - self.d) / (self.u - self.d)
        
#         prc = 0
#         temp_stock = 0
#         temp_payout = 0
#         for x in range(0, self.n + 1):
#             temp_stock = self.s0*((self.u)**(x))*((self.d)**(self.n - x))
#             temp_payout = max(temp_stock - self.strike, 0)
#             prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
#         if self.compd == "s":
#             prc = prc / ((1+ self.rfr*delta )**self.n)
#         if self.compd == "c":
#             prc = prc / math.exp(self.rfr*delta)
        
        
#         return prc
    
    
#     def put_price(self):
#         delta = float(self.maturity)/float(self.n)
        
#         if self.compd == "c":
#             if self.dyield ==None: 
#                 q = (math.exp(self.rfr*delta) - self.d) / (self.u - self.d)
#             else:
#                 q = (math.exp((self.rfr-self.dyield)*delta) - self.d) / (self.u - self.d)
#         if self.compd == "s":
#             if self.dyield == None: 
#                 q = (1 + self.rfr*delta - self.d) / (self.u - self.d)
#             else:
#                 q = (1+ (self.rfr - self.dyield)*delta - self.d) / (self.u - self.d)
        
#         prc = 0
#         temp_stock = 0
#         temp_payout = 0
#         for x in range(0, self.n + 1):
#             temp_stock = self.s0*((self.u)**(x))*((self.d)**(self.n - x))
#             temp_payout = max(self.strike - temp_stock, 0)
#             prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
#         if self.compd == "s":
#             prc = prc / ((1+ self.rfr*delta )**self.n)
#         if self.compd == "c":
#             prc = prc / math.exp(self.rfr*delta)
        
        
#         return prc

def BinomialModel(PutCall, n, S0, X, rfr, u, d, t, AMNEUR, compd='s', dyield=None):
    deltaT = t / n
    if compd == "c":
        if dyield is None:
            p = (np.exp(rfr * deltaT) - d) / (u - d)
        else:
            p = (np.exp((rfr - dyield) * deltaT) - d) / (u - d)
    elif compd == "s":
        if dyield is None:
            p = (1 + rfr * deltaT - d) / (u - d)
        else:
            p = (1 + (rfr - dyield) * deltaT - d) / (u - d)
    else:
        raise ValueError("Invalid value for compd. Use 'c' or 's'.")

    q = 1 - p
    
    # Simulating the underlying price paths
    S = np.zeros((n + 1, n + 1))
    S[0, 0] = S0
    for i in range(1, n + 1):
        for j in range(i + 1):
            S[i, j] = S0 * (u ** j) * (d ** (i - j))
    
    # Option value at final node
    V = np.zeros((n + 1, n + 1))
    
    for j in range(n + 1):
        if PutCall == "C":
            V[n, j] = max(0, S[n, j] - X)
        elif PutCall == "P":
            V[n, j] = max(0, X - S[n, j])
    
    # European Option: backward induction to the option price V[0, 0]
    if AMNEUR == "E":
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                V[i, j] = np.exp(-rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1])
        opt_price = V[0, 0]
    
    # American Option: backward induction to the option price V[0, 0]
    elif AMNEUR == "A":
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                if PutCall == "P":
                    V[i, j] = max(0, X - S[i, j], np.exp(-rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1]))
                elif PutCall == "C":
                    V[i, j] = max(0, S[i, j] - X, np.exp(-rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1]))
        opt_price = V[0, 0]
    
    return opt_price

# class CRRModel(object):
#     def __init__(self, s0, sigma, strike, maturity, rfr,  n, compd = "s", dyield = None):      
#         self.s0 = s0 
#         self.sigma = sigma
#         self.rfr = rfr
#         self.maturity = maturity 
#         self.strike = strike
#         self.n = n
#         self.compd = compd
#         self.dyield = dyield
    
#     def call_price(self):
#         delta = float(self.maturity)/float(self.n)
#         u = math.exp(self.sigma*math.sqrt(delta))
#         d = 1/math.exp(self.sigma*math.sqrt(delta))


#         if self.compd == "c":
#             if self.dyield ==None: 
#                 q = (math.exp(self.rfr*delta) - d) / (u - d)
#             else:
#                 q = (math.exp((self.rfr-self.dyield)*delta) - d) / (u - d)
#         if self.compd == "s":
#             if self.dyield == None: 
#                 q = (1 + self.rfr*delta - d) / (u - d)
#             else:
#                 q = (1+ (self.rfr - self.dyield)*delta - d) / (u - d)
        
#         prc = 0
#         temp_stock = 0
#         temp_payout = 0
#         for x in range(0, self.n + 1):
#             temp_stock = self.s0*((u)**(x))*((d)**(self.n - x))
#             temp_payout = max(temp_stock - self.strike, 0)
#             prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
#         if self.compd == "s":
#             prc = prc / ((1+ self.rfr*delta )**self.n)
#         if self.compd == "c":
#             prc = prc / math.exp(self.rfr*delta)
        
        
#         return prc
    
    
#     def put_price(self):
#         delta = float(self.maturity)/float(self.n)
#         u = math.exp(self.sigma*math.sqrt(delta))
#         d= 1/math.exp(self.sigma*math.sqrt(delta))


#         if self.compd == "c":
#             if self.dyield ==None: 
#                 q = (math.exp(self.rfr*delta) - d) / (u - d)
#             else:
#                 q = (math.exp((self.rfr-self.dyield)*delta) - d) / (u - d)
#         if self.compd == "s":
#             if self.dyield == None: 
#                 q = (1 + self.rfr*delta - d) / (u - d)
#             else:
#                 q = (1+ (self.rfr - self.dyield)*delta - d) / (u - d)
        
#         prc = 0
#         temp_stock = 0
#         temp_payout = 0
#         for x in range(0, self.n + 1):
#             temp_stock = self.s0*((u)**(x))*((d)**(self.n - x))
#             temp_payout = max(self.strike - temp_stock, 0)
#             prc += nCr(self.n, x)*(q**(x))*((1-q)**(self.n - x))*temp_payout
        
#         if self.compd == "s":
#             prc = prc / ((1+ self.rfr*delta )**self.n)
#         if self.compd == "c":
#             prc = prc / math.exp(self.rfr*delta)
        
        
#         return prc  

def CRRModel(PutCall, n, S0, X, rfr, vol, t, AMNEUR, dyield=None, cmpd="s"):
    deltaT = t / n
    u = np.exp(vol * np.sqrt(deltaT))
    d = 1.0 / u
    
    if cmpd == "c":
        if dyield is None:
            p = (np.exp(rfr * deltaT) - d) / (u - d)
        else:
            p = (np.exp((rfr - dyield) * deltaT) - d) / (u - d)
    elif cmpd == "s":
        if dyield is None:
            p = (1 + rfr * deltaT - d) / (u - d)
        else:
            p = (1 + (rfr - dyield) * deltaT - d) / (u - d)
    else:
        raise ValueError("Invalid value for cmpd. Use 'c' or 's'.")
    
    q = 1 - p
    
    # Simulating the underlying price paths
    S = np.zeros((n + 1, n + 1))
    S[0, 0] = S0
    for i in range(1, n + 1):
        S[i, 0] = S[i - 1, 0] * u
        for j in range(1, i + 1):
            S[i, j] = S[i - 1, j - 1] * d
    
    # Option value at final node
    V = np.zeros((n + 1, n + 1))
    
    for j in range(n + 1):
        if PutCall == "C":
            V[n, j] = max(0, S[n, j] - X)
        elif PutCall == "P":
            V[n, j] = max(0, X - S[n, j])
    
    # European Option: backward induction to the option price V[0, 0]
    if AMNEUR == "E":
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                V[i, j] = max(0, 1 / (1 + rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1]))
        opt_price = V[0, 0]
    
    # American Option: backward induction to the option price V[0, 0]
    elif AMNEUR == "A":
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                if PutCall == "P":
                    V[i, j] = max(0, X - S[i, j], 1 / (1 + rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1]))
                elif PutCall == "C":
                    V[i, j] = max(0, S[i, j] - X, 1 / (1 + rfr * deltaT) * (p * V[i + 1, j] + q * V[i + 1, j + 1]))
        opt_price = V[0, 0]
    
    return opt_price

# class BlackScholeModel(object):
#     def __init__(self, S, K, T, r, sigma, q=0,):
#         self.S = S
#         self.K = K
#         self.T = T
#         self.r = r
#         self.sigma = sigma
#         self.q = q
    
#     @staticmethod
#     def N(x):
#         return norm.cdf(x)
    
#     @property
#     def params(self):
#         return {'S': self.S, 
#                 'K': self.K, 
#                 'T': self.T, 
#                 'r': self.r,
#                 'q': self.q,
#                 'sigma': self.sigma}
    
#     def d1(self):
#         return (np.log(self.S / self.K) + (self.r - self.q + self.sigma**2 / 2) * self.T) \
#                                 / (self.sigma * np.sqrt(self.T))
    
#     def d2(self):
#         return self.d1() - self.sigma * np.sqrt(self.T)
    
#     def call_value(self):
#         return self.S * np.exp(-self.q * self.T) * self.N(self.d1()) - \
#                     self.K * np.exp(-self.r * self.T) * self.N(self.d2())
                    
#     def put_value(self):
#         return self.K * np.exp(-self.r * self.T) * self.N(-self.d2()) - \
#                 self.S * np.exp(-self.q * self.T) * self.N(-self.d1())
    
#     # def price(self, type_='C'):
#     #     if type_ == 'C':
#     #         return self._call_value()
#     #     if type_ == 'P':
#     #         return self._put_value() 
#     #     if type_ == 'B':
#     #         return {'call': self._call_value(), 'put': self._put_value()}
#     #     else:
#     #         raise ValueError('Unrecognized type')

def BlackScholeModel(PutCall, S0, X, rfr, vol, t):
    d1 = (np.log(S0 / X) + (rfr + vol ** 2 / 2) * t) / (vol * np.sqrt(t))
    d2 = d1 - vol * np.sqrt(t)
    
    if PutCall == "C":
        opt_price = S0 * norm.cdf(d1) - X * np.exp(-rfr * t) * norm.cdf(d2)
    elif PutCall == "P":
        opt_price = X * np.exp(-rfr * t) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    
    return opt_price

def getPriceAndProbBS(request):
    if request.method=='POST':
        initialEP=float(request.POST.get('val'))
        Volatility=float(request.POST.get('volatility')) 
        strikePrice=float(request.POST.get('strikePrice'))
        riskFreeRate=float(request.POST.get('riskFreeRate'))
        maturity=float(request.POST.get('maturity'))
        dYield=float(request.POST.get('dYield'))
        isPut=request.POST.get('isPut')  

        if isPut=='true':
            fairPrice=BlackScholeModel('P',initialEP,strikePrice,riskFreeRate,Volatility,maturity)  
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=BlackScholeModel('C',initialEP,strikePrice,riskFreeRate,Volatility,maturity) 
            return JsonResponse({'fairPrice':round(callFairPrice,2)})

def getPriceAndProb(request):
    if request.method=='POST':
        initialEP=float(request.POST.get('val'))
        upFactor=float(request.POST.get('upFactor'))
        downFactor=float(request.POST.get('downFactor'))
        strikePrice=float(request.POST.get('strikePrice'))
        riskFreeRate=float(request.POST.get('riskFreeRate'))
        maturity=float(request.POST.get('maturity'))
        noOfPeriods=int(request.POST.get('noOfPeriods'))
        dYield=float(request.POST.get('dYield'))
        com=(request.POST.get('interest'))
        isPut=request.POST.get('isPut') 

        if isPut=='true':            
            fairPrice=BinomialModel('P',noOfPeriods,initialEP,strikePrice,riskFreeRate,upFactor,downFactor,maturity,'E',com,dYield)
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=BinomialModel('C',noOfPeriods,initialEP,strikePrice,riskFreeRate,upFactor,downFactor,maturity,'E',com,dYield)
            return JsonResponse({'fairPrice':round(callFairPrice,2)})

def getPriceAndProbCRR(request):
    if request.method=='POST':
        initialEP=float(request.POST.get('val'))
        Volatility=float(request.POST.get('volatility')) 
        strikePrice=float(request.POST.get('strikePrice'))
        riskFreeRate=float(request.POST.get('riskFreeRate'))
        maturity=float(request.POST.get('maturity'))
        noOfPeriods=int(request.POST.get('noOfPeriods'))
        dYield=float(request.POST.get('dYield'))
        com=(request.POST.get('interest'))
        isPut=request.POST.get('isPut')
       
        if isPut=='true':
            fairPrice=CRRModel('P',noOfPeriods,initialEP,strikePrice,riskFreeRate,Volatility,maturity,'E',dYield,com)
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=CRRModel('C',noOfPeriods,initialEP,strikePrice,riskFreeRate,Volatility,maturity,'E',dYield,com)
            return JsonResponse({'fairPrice':round(callFairPrice,2)})

def home(req):
    return render(req,'home.html',{})