## 1900 vs 1930 vs 1950 vs 1980 vs 2018 – scatter plot – label each country with colour according subregion
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

GDP_CE = np.genfromtxt('GDP_CE.csv', delimiter=',')
GDP_Northern = np.genfromtxt('GDP_Northern.csv', delimiter=',')
GDP_Southern = np.genfromtxt('GDP_Southern.csv', delimiter=',')
GDP_Western = np.genfromtxt('GDP_Western.csv', delimiter=',')

LE_CE = np.genfromtxt('LE_CE.csv', delimiter=',')
LE_Northern = np.genfromtxt('LE_Northern.csv', delimiter=',')
LE_Southern = np.genfromtxt('LE_Southern.csv', delimiter=',')
LE_Western = np.genfromtxt('LE_Western.csv', delimiter=',')

#Year: 1900

#GDP @1900
GDP_CE_1900 = GDP_CE[0,1:]
GDP_Northern_1900 = GDP_Northern[0,1:]
GDP_Southern_1900 = GDP_Southern[0,1:]
GDP_Western_1900= GDP_Western[0,1:]


#LE @1900
LE_CE_1900 = LE_CE[0,1:]
LE_Northern_1900 = LE_Northern[0,1:]
LE_Southern_1900 = LE_Southern[0,1:]
LE_Western_1900 = LE_Western[0,1:]

#Scatter plot
plt.scatter(GDP_CE_1900, LE_CE_1900, color = 'red')
plt.scatter(GDP_Northern_1900, LE_Northern_1900, color = 'mediumblue')
plt.scatter(GDP_Southern_1900, LE_Southern_1900, color = 'gold')
plt.scatter(GDP_Western_1900, LE_Western_1900, color = 'limegreen')

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')
plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy in 1900')
plt.grid()

plt.savefig('Income vs LE - 1900.png', dpi=800)

#Year: 1930

#GDP @1930
GDP_CE_1930 = GDP_CE[30,1:]
GDP_Northern_1930 = GDP_Northern[30,1:]
GDP_Southern_1930 = GDP_Southern[30,1:]
GDP_Western_1930= GDP_Western[30,1:]


#LE @1930
LE_CE_1930 = LE_CE[30,1:]
LE_Northern_1930 = LE_Northern[30,1:]
LE_Southern_1930 = LE_Southern[30,1:]
LE_Western_1930 = LE_Western[30,1:]

#Scatter plot
plt.scatter(GDP_CE_1930, LE_CE_1930, color = 'red')
plt.scatter(GDP_Northern_1930, LE_Northern_1930, color = 'mediumblue')
plt.scatter(GDP_Southern_1930, LE_Southern_1930, color = 'gold')
plt.scatter(GDP_Western_1930, LE_Western_1930, color = 'limegreen')

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')
plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy in 1930')
plt.grid()

plt.savefig('Income vs LE - 1930.png', dpi=800)

#Year: 1950

#GDP @1950
GDP_CE_1950 = GDP_CE[50,1:]
GDP_Northern_1950 = GDP_Northern[50,1:]
GDP_Southern_1950 = GDP_Southern[50,1:]
GDP_Western_1950 = GDP_Western[50,1:]


#LE @1950
LE_CE_1950 = LE_CE[50,1:]
LE_Northern_1950 = LE_Northern[50,1:]
LE_Southern_1950 = LE_Southern[50,1:]
LE_Western_1950 = LE_Western[50,1:]

#Scatter plot
plt.scatter(GDP_CE_1950, LE_CE_1950, color = 'red')
plt.scatter(GDP_Northern_1950, LE_Northern_1950, color = 'mediumblue')
plt.scatter(GDP_Southern_1950, LE_Southern_1950, color = 'gold')
plt.scatter(GDP_Western_1950, LE_Western_1950, color = 'limegreen')

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')
plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy in 1950')
plt.grid()

plt.savefig('Income vs LE - 1950.png', dpi=800)

#Year: 1980

#GDP @1980
GDP_CE_1980 = GDP_CE[80,1:]
GDP_Northern_1980 = GDP_Northern[80,1:]
GDP_Southern_1980 = GDP_Southern[80,1:]
GDP_Western_1980 = GDP_Western[80,1:]


#LE @1980
LE_CE_1980 = LE_CE[80,1:]
LE_Northern_1980 = LE_Northern[80,1:]
LE_Southern_1980 = LE_Southern[80,1:]
LE_Western_1980 = LE_Western[80,1:]

#Scatter plot
plt.scatter(GDP_CE_1980, LE_CE_1980, color = 'red')
plt.scatter(GDP_Northern_1980, LE_Northern_1980, color = 'mediumblue')
plt.scatter(GDP_Southern_1980, LE_Southern_1980, color = 'gold')
plt.scatter(GDP_Western_1980, LE_Western_1980, color = 'limegreen')

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')
plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy in 1980')
plt.grid()

plt.savefig('Income vs LE - 1980.png', dpi=800)

#Year: 2018

#GDP @2018
GDP_CE_2018 = GDP_CE[118,1:]
GDP_Northern_2018 = GDP_Northern[118,1:]
GDP_Southern_2018 = GDP_Southern[118,1:]
GDP_Western_2018 = GDP_Western[118,1:]

#Life Expectancy (LE) @2018
LE_CE_2018 = LE_CE[118,1:]
LE_Northern_2018 = LE_Northern[118,1:]
LE_Southern_2018 = LE_Southern[118,1:]
LE_Western_2018 = LE_Western[118,1:]

#Scatter plot
plt.scatter(GDP_CE_2018, LE_CE_2018, color = 'red')
plt.scatter(GDP_Northern_2018, LE_Northern_2018, color = 'mediumblue')
plt.scatter(GDP_Southern_2018, LE_Southern_2018, color = 'gold')
plt.scatter(GDP_Western_2018, LE_Western_2018, color = 'limegreen')

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')
plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy in 2018')
plt.grid()

plt.savefig('Income vs LE - 2018.png', dpi=800)
