#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 23 17:01:48 2021

@author: zhongyizhou
"""

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web

#this program choose flattener strategy, 10Y-2Y DV01 neutral spread trade, reblanced weekly
#inport data from csv file
rawDF = pd.read_csv('gsw_yields_2021.csv')
c_name = rawDF.iloc[8,:].values.tolist()

#initialize dataframe row and column names
rawDF.set_axis(c_name,axis=1,inplace=True)
rawDF = rawDF.iloc[9:,:].set_index('Date')

#convert index to datetimeindex
rawDF.index=pd.to_datetime(rawDF.index)

#select targeted date range
start_date = '1983-12-30'
end_date = '2021-06-30'
targetDF = rawDF[(rawDF.index>=start_date) & (rawDF.index<=end_date)]

#resample from daily to weekly
weekDF=targetDF.resample('W').last()
loffset = targetDF.index.dayofweek

#need to offset dateindex in weekDF
#weekDF.index = [i - dt.timedelta(j) for i, j in zip(weekDF.index, (6- loffset))]
#####################################################################################

#####################################################################################
#function to extrapolate ytm 
#input: y: maturity in years
#extrapolate 9Y358D and 2Y358D rates
def extrapolate (y):
    t_1 = y/pd.to_numeric(weekDF['TAU1'])
    t_2 = y/pd.to_numeric(weekDF['TAU2'])
    exp_t_1 = np.exp(-t_1)
    exp_t_2 = np.exp(-t_2)
    r_y = pd.to_numeric(weekDF['BETA0'])+ pd.to_numeric(weekDF['BETA1'])*(1-exp_t_1)/t_1+\
        pd.to_numeric(weekDF['BETA2'])*((1-exp_t_1)/t_1-exp_t_1)+\
            pd.to_numeric(weekDF['BETA3'])*((1-exp_t_2)/t_2-exp_t_2)
    return r_y

#####################################################################################


#calculate 2y and 10y zero bond price
two_ytm = pd.to_numeric(weekDF['SVENY02'])/100
ten_ytm = pd.to_numeric(weekDF['SVENY10'])/100
#####################################################################################
#input: margin: in percentage(%), capital: initial capital
def flattener(margin, capital):
    #creat dataframe to hold info
    priceDF = pd.DataFrame(index= weekDF.index)

    #calculate 2y and 10y zero bond price
    #two_ytm = pd.to_numeric(weekDF['SVENY02'])/100
    #ten_ytm = pd.to_numeric(weekDF['SVENY10'])/100

    priceDF['2Y_Price'] = 100*np.exp(-two_ytm*2)
    priceDF['10Y_Price'] = 100*np.exp(-ten_ytm*10)
    #priceDF['2Y_Price'] = 100/(1+two_ytm)**2
    #priceDF['10Y_Price'] = 100/(1+ten_ytm)**10

    #calculate 2y and 10y zero bond DV01
    priceDF['2Y_DV01'] = (2/(1+two_ytm))*priceDF['2Y_Price']/10000
    priceDF['10Y_DV01'] = (10/(1+ten_ytm))*priceDF['10Y_Price']/10000
    
    
    #1 week yield
    priceDF['1W_rate'] = extrapolate(7/365)/100
    
    #new yield for 10Y and 2 Y
    priceDF['9Y_358D_ytm'] = extrapolate(9+358/365)/100
    priceDF['1Y_358D_ytm'] = extrapolate(1+358/365)/100
    #price at rebalance        
    priceDF['9Y_358D'] = 100*np.exp(-priceDF['9Y_358D_ytm']*(9+358/365))
    priceDF['1Y_358D'] = 100*np.exp(-priceDF['1Y_358D_ytm']*(1+358/365))
    startFund = capital
    
    priceDF['begin_value'] = startFund
    
    #calulate hedge ratio and position
    priceDF['h_ratio']=priceDF['2Y_DV01']/priceDF['10Y_DV01']
    #priceDF['2Y_Q']=startFund*10/(-priceDF['10Y_DV01']*priceDF['h_ratio']+priceDF['2Y_DV01'])
    #priceDF['10Y_Q']=-priceDF['2Y_Q']*priceDF['h_ratio']    
    
    #calculate trade statistics
    #2Y position
    priceDF['2Y_Q']=""
    #10Y position
    priceDF['10Y_Q']=""
    #cash position
    priceDF['cash'] = ""
    #interest received/paid in dollar
    priceDF['int_yield'] = ""
    #total return
    priceDF['tot_rtn'] = ""

    #update priceDF
    for index in priceDF.index:
        if (index==priceDF.index[0]):
            #if the first day, only open positions, and calculate cash and interest received/paid
            #calculation 2Y and 10Y positions
            #2Y_Q and 10Y_Q are object, need to check
            priceDF.loc[index,'2Y_Q']=-(startFund/margin)/(priceDF.loc[index,'10Y_Price']*priceDF.loc[index,'h_ratio']+priceDF.loc[index,'2Y_Price'])
            priceDF.loc[index,'10Y_Q']=-priceDF.loc[index,'2Y_Q']*priceDF.loc[index,'h_ratio']
            #caculate cash position
            priceDF.loc[index,'cash'] = -priceDF.loc[index,'2Y_Q']*priceDF.loc[index,'2Y_Price'] -\
                priceDF.loc[index,'10Y_Q']*priceDF.loc[index,'10Y_Price']+startFund
            #calculate interest received/paid
            priceDF.loc[index,'int_yield'] = priceDF.loc[index,'cash']*(np.exp(priceDF.loc[index,'1W_rate']*7/365)-1)
            priceDF.loc[index,'tot_rtn'] = 0
        else:
            #close previous positions
            #sell old positions plus/minus any interest received/paid
            temp = priceDF.shift(1)
            priceDF.loc[index,'begin_value'] = temp.loc[index,'cash'] + \
                temp.loc[index,'2Y_Q']*priceDF.loc[index,'1Y_358D'] +temp.loc[index,'10Y_Q']*priceDF.loc[index,'9Y_358D']\
                    +temp.loc[index,'int_yield']
            priceDF.loc[index,'tot_rtn'] = priceDF.loc[index,'begin_value']/startFund-1
            if (index != priceDF.index[-1]):
            #only open new positions if not the last week
            #calculate new positions
                priceDF.loc[index,'2Y_Q']=-(priceDF.loc[index,'begin_value']/margin)/(priceDF.loc[index,'10Y_Price']*priceDF.loc[index,'h_ratio']+priceDF.loc[index,'2Y_Price'])
                priceDF.loc[index,'10Y_Q']=-priceDF.loc[index,'2Y_Q']*priceDF.loc[index,'h_ratio']
            #caculate cash position
                priceDF.loc[index,'cash'] = -priceDF.loc[index,'2Y_Q']*priceDF.loc[index,'2Y_Price'] -\
                    priceDF.loc[index,'10Y_Q']*priceDF.loc[index,'10Y_Price']+priceDF.loc[index,'begin_value']
            #calculate interest received/paid
                priceDF.loc[index,'int_yield'] = priceDF.loc[index,'cash']*(np.exp(priceDF.loc[index,'1W_rate']*7/365)-1)
    #priceDF[['tot_rtn']].plot()
    return priceDF
#####################################################################################

priceDF2 = flattener(0.1, 1000000)
#get historical 10Y-2Y spread from fred
his_spread = web.get_data_fred('T10Y2Y', start_date, end_date).dropna()
#plot return and spread
fig,ax = plt.subplots()
ax.plot(priceDF2['tot_rtn'], color="red")
ax.set_ylabel("Flattener Return",color="red")
ax2=ax.twinx()
ax2.plot(his_spread['T10Y2Y'],color="green")
ax2.set_ylabel("Spread",color="green")
plt.show()

'''
plt.plot(his_spread['T10Y2Y'], label='Spread', color='green')
plt.plot(priceDF['tot_rtn'], label='Return', color='red')
plt.show()
'''

'''
#calculate yield on each trade
priceDF['yield'] = ""
for i in range(len(priceDF)-1):
    priceDF.iloc[i+1,priceDF.columns.get_loc('yield')]=(priceDF.iloc[i,priceDF.columns.get_loc('2Y_DV01')]-\
        priceDF.iloc[i+1,priceDF.columns.get_loc('2Y_DV01')])*priceDF.iloc[i,priceDF.columns.get_loc('2Y_Q')]+\
        (priceDF.iloc[i+1,priceDF.columns.get_loc('10Y_DV01')]-\
        priceDF.iloc[i,priceDF.columns.get_loc('10Y_DV01')])*priceDF.iloc[i,priceDF.columns.get_loc('10Y_Q')]
    i+=1

priceDF.iloc[0,priceDF.columns.get_loc('yield')]=0
priceDF['cum_return']=np.cumsum(priceDF['yield']/startFund)
'''

#plt.plot(range(len(priceDF)),priceDF['cum_return'])
#####################################################################################
#part2
part2DF = pd.DataFrame()
part2DF['CONV10'] = 10*(10+1)/(1+ten_ytm)**2
part2DF['10Y_Price'] = priceDF2['10Y_Price']
#10 bp change
yield_change = 0.001
part2DF['CON_Risk'] = 0.5*10000*part2DF['10Y_Price']*part2DF['CONV10']*yield_change**2
plt.plot(part2DF['CON_Risk'], label='Spread', color='green')


#####################################################################################
#part3

part2DF['begin_value'] = priceDF2['begin_value']
part2DF['tot_rtn'] = priceDF2['tot_rtn']
part2DF['10_2_Spread'] = (pd.to_numeric(weekDF['SVENY10']) -pd.to_numeric(weekDF['SVENY02']))/100
part2DF['9_1_Spread'] = priceDF2['9Y_358D_ytm'] - priceDF2['1Y_358D_ytm']
part2DF['tot_DV01'] = (priceDF2['2Y_Price']* pd.to_numeric(priceDF2['2Y_Q'])*priceDF2['2Y_DV01']+\
                       priceDF2['10Y_Price']* pd.to_numeric(priceDF2['10Y_Q'])*priceDF2['10Y_DV01'])/priceDF2['begin_value']
part2DF['CONV2']=2*(2+1)/(1+two_ytm)**2
part2DF['V2']= priceDF2['2Y_Price']*pd.to_numeric(priceDF2['2Y_Q'])
part2DF['V10']= priceDF2['10Y_Price']*pd.to_numeric(priceDF2['10Y_Q'])
#part2DF['CVX_10']=10*(10+1)/(1+ten_ytm)**2

part2DF['Sp_rtn'] = ""
part2DF['Cvx_rtn'] = ""
part2DF['dy_2'] = priceDF2['1Y_358D_ytm']-pd.to_numeric(weekDF.shift(1)['SVENY02'])/100
part2DF['dy_10'] = priceDF2['9Y_358D_ytm']-pd.to_numeric(weekDF.shift(1)['SVENY10'])/100
for index in part2DF.index:
    if index == part2DF.index[0]:
        part2DF.loc[index,'Sp_rtn'] = 0
        part2DF.loc[index,'Cvx_rtn'] = 0
    else:
        temp = part2DF.shift(1)
        part2DF.loc[index,'Sp_rtn'] = temp.loc[index, 'begin_value']*(part2DF.loc[index,'9_1_Spread']- temp.loc[index,'10_2_Spread'])*part2DF.loc[index,'tot_DV01']
        #p2 =  
        part2DF.loc[index,'Cvx_rtn'] = 0.5*temp.loc[index, 'CONV2']*temp.loc[index,'V2']*part2DF.loc[index,'dy_2']**2 +\
            0.5*temp.loc[index, 'CONV10']*temp.loc[index,'V10']*part2DF.loc[index,'dy_10']**2

plt.plot(part2DF['Sp_rtn'], label='Sp_rtn', color='green')  
plt.plot(part2DF['Cvx_rtn'], label='Sp_rtn', color='green')  
#cvxt1_358 =  (1+358/365)*(2+358/365)/(1+priceDF['1Y_358D_ytm'])**2
#cvxt9_358 = (9+358/365)*(10+358/365)/(1+priceDF['1Y_358D_ytm'])**2
    
'''
#3b
#calculate ZCB convexity
cvxt2=2*(2+1)/(1+two_ytm)**2
cvxt10=10*(10+1)/(1+ten_ytm)**2
D2=2/(1+two_ytm)
D10=2/(1+ten_ytm)
priceDF['2Y_yieldchg']=two_ytm.diff()
priceDF['2Y_pricechg']=-priceDF['2Y_Price']*D2*priceDF['2Y_yieldchg']+1/2*priceDF['2Y_Price']*cvxt2*priceDF['2Y_yieldchg']**2
priceDF['10Y_yieldchg']=ten_ytm.diff()
priceDF['10Y_pricechg']=-priceDF['10Y_Price']*D10*priceDF['10Y_yieldchg']+1/2*priceDF['10Y_Price']*cvxt10*priceDF['10Y_yieldchg']**2
priceDF['cvxtreturn']=-priceDF['2Y_Q']*priceDF['2Y_pricechg']+priceDF['10Y_Q']*priceDF['10Y_pricechg']
priceDF['cum_cvxtretreturn']=np.cumsum(priceDF['cvxtreturn'])
plt.plot(priceDF['cum_cvxtretreturn'])
'''

