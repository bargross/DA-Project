## Life expectancy vs income – 1900 – 2018 – all 4 regions in plot

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

GDP_Europe = np.genfromtxt('AVG_GDP_Europe.csv', delimiter =',')
LE_Europe = np.genfromtxt('AVG_Life_expectancy_Europe.csv', delimiter=',')

GDP_CE_av = GDP_Europe[2:,1]
GDP_N_av = GDP_Europe[2:,2]
GDP_S_av = GDP_Europe[2:,3]
GDP_W_av = GDP_Europe[2:,4]

LE_CE_av = LE_Europe[2:,1]
LE_N_av = LE_Europe[2:,2]
LE_S_av = LE_Europe[2:,3]
LE_W_av = LE_Europe[2:,4]


plt.scatter(GDP_CE_av, LE_CE_av, color = 'red', s=8)
plt.scatter(GDP_N_av, LE_N_av, color = 'mediumblue', s=8)
plt.scatter(GDP_S_av, LE_S_av, color = 'gold', s=8)
plt.scatter(GDP_W_av, LE_W_av, color = 'limegreen', s=8)

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')

plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income vs Life expectancy during 1900-2018')
plt.grid()

plt.savefig('Preston curve (scatter) - Europe.png', dpi=800)

## Life expectancy vs income – 1900 – 2018 – all 4 regions in plot


GDP_Europe = np.genfromtxt('AVG_GDP_Europe.csv', delimiter =',')
LE_Europe = np.genfromtxt('AVG_Life_expectancy_Europe.csv', delimiter=',')

GDP_CE_av = GDP_Europe[2:,1]
GDP_N_av = GDP_Europe[2:,2]
GDP_S_av = GDP_Europe[2:,3]
GDP_W_av = GDP_Europe[2:,4]

LE_CE_av = LE_Europe[2:,1]
LE_N_av = LE_Europe[2:,2]
LE_S_av = LE_Europe[2:,3]
LE_W_av = LE_Europe[2:,4]



coefs_CE = np.polyfit(GDP_CE_av, LE_CE_av, 4)
print(coefs_CE)
ce = np.poly1d(coefs_CE)

coefs_N = np.polyfit(GDP_N_av, LE_N_av, 6)
print(coefs_N)
n = np.poly1d(coefs_N)

coefs_S = np.polyfit(GDP_S_av, LE_S_av, 6)
print(coefs_S)
s = np.poly1d(coefs_S)

coefs_W = np.polyfit(GDP_W_av, LE_W_av, 5)
print(coefs_W)
w = np.poly1d(coefs_W)


plt.plot(GDP_CE_av, ce(GDP_CE_av), 'red')
plt.plot(GDP_N_av, n(GDP_N_av), 'mediumblue')
plt.plot(GDP_S_av, s(GDP_S_av), 'gold')
plt.plot(GDP_W_av, w(GDP_W_av), 'limegreen')

plt.grid()

plt.xlabel('GDP per capita (USD)')
plt.ylabel('Life expectancy')

plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('The Preston curve in Europe, during 1900-2018')

plt.savefig('The Preston curve - Europe.png', dpi=800)
