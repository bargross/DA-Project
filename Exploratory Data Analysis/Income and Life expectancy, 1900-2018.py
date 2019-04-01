##To be plotted:
## life expectancy over time – all 4 regions in a plot
## GDP over time – all 4 regions in a plot

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

## life expectancy over time

LE_data = np.genfromtxt('AVG_Life_expectancy_Europe.csv', delimiter=',')

year = LE_data[:, 0]
LE_CE = LE_data[:, 1]
LE_N = LE_data[:, 2]
LE_S = LE_data[:, 3]
LE_W = LE_data[:, 4]

plt.plot(year, LE_CE, color = 'red')
plt.plot(year, LE_N, color = 'mediumblue')
plt.plot(year, LE_S, color = 'gold')
plt.plot(year, LE_W, color = 'limegreen')

plt.xlabel('Year')
plt.ylabel('Life expectancy')

plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Life expectancy in Europe, 1900-2018')
plt.grid()

plt.savefig('Life expectancy over time.png', dpi=800)

## GDP over time

GDP_data = np.genfromtxt('AVG_GDP_Europe.csv', delimiter=',')

year = GDP_data[:, 0]
GDP_CE = GDP_data[:, 1]
GDP_N = GDP_data[:, 2]
GDP_S = GDP_data[:, 3]
GDP_W = GDP_data[:, 4]

plt.plot(year, GDP_CE, color = 'red')
plt.plot(year, GDP_N, color = 'mediumblue')
plt.plot(year, GDP_S, color = 'gold')
plt.plot(year, GDP_W, color = 'limegreen')

plt.xlabel('Year')
plt.ylabel('GDP per capita (USD)')

plt.legend(['Eastern & Central', 'Northern', 'Southern', 'Western'])
plt.title('Income (GDP/capita) in Europe, 1900-2018')
plt.grid()

plt.savefig('GDP over time.png', dpi=800)
