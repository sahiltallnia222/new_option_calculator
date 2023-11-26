from django.shortcuts import render,redirect
import math
import yfinance as yf
from django.http import JsonResponse 
from pandas_datareader import data, wb
import matplotlib.pyplot as plt
import pandas_datareader as pdr
from scipy.stats import norm
import numpy as np    

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)
 
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



def CRRModel(PutCall, n, S0, X, rfr, vol, t,compSymBol, AMNEUR, dyield=None, cmpd="s"):
    print(AMNEUR)
    stock_data = yf.Ticker(compSymBol)
    data = stock_data.history()
    close_prices = data['Close']
    daily_returns = close_prices.pct_change().dropna()
    vol = daily_returns.std()*np.sqrt(252)
    
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

def BlackScholeModel(PutCall, S0, X, rfr, vol, t,cSymbol):
    # ticker = "GOOG"  # Replace with the desired stock symbol
    # start_date = "2021-01-01"
    # end_date = "2021-12-31"
    # print(cSymbol)
    stock_data = yf.Ticker(cSymbol)
    data = stock_data.history()
    close_prices = data['Close']
    daily_returns = close_prices.pct_change().dropna()
    vol = daily_returns.std()*np.sqrt(252)
    # print(vol)                                                                                              
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
        cSymbol=request.POST.get('cSymbol')
        # print(cSymbol)
        if isPut=='true':
            fairPrice=BlackScholeModel('P',initialEP,strikePrice,riskFreeRate,Volatility,maturity,cSymbol)  
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=BlackScholeModel('C',initialEP,strikePrice,riskFreeRate,Volatility,maturity,cSymbol) 
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
        selectedVal=request.POST.get('selectedVal')
        if isPut=='true':            
            fairPrice=BinomialModel('P',noOfPeriods,initialEP,strikePrice,riskFreeRate,upFactor,downFactor,maturity,selectedVal,com,dYield)
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=BinomialModel('C',noOfPeriods,initialEP,strikePrice,riskFreeRate,upFactor,downFactor,maturity,selectedVal,com,dYield)
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
        compSymBol=request.POST.get('compSymBolCRR')
        selectedValCRR=request.POST.get('selectedValCRR')
        if isPut=='true':
            fairPrice=CRRModel('P',noOfPeriods,initialEP,strikePrice,riskFreeRate,Volatility,maturity,compSymBol,selectedValCRR,dYield,com)
            return JsonResponse({'fairPrice':round(fairPrice,2)})
        else:
            callFairPrice=CRRModel('C',noOfPeriods,initialEP,strikePrice,riskFreeRate,Volatility,maturity,compSymBol,selectedValCRR,dYield,com)
            return JsonResponse({'fairPrice':round(callFairPrice,2)})

def home(req):
    return render(req,'home.html',{})