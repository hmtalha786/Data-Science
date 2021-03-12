#!/usr/bin/env python
# coding: utf-8

# ## Matplotlib Tutorial

# ### Imports

# In[ ]:


# Import Matplotlib and allow plots to show in the Jupyter Notebook
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Import NumPy and Pandas
import numpy as np
import pandas as pd


# In[ ]:


# Auto reloads notebook when changes are made
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# ### Functional Plot

# In[ ]:


# Create x & y NumPy arrays (10 Floats from 0 to 5)
# Any NumPy array will do for x & y points as long as you have 
# an equal number 
x_1 = np.linspace(0,5,10)
y_1 = x_1**2


# In[ ]:


# Display plot with x & y
plt.plot(x_1, y_1)


# In[ ]:


# Add a title
plt.title('Days Squared Chart')


# In[ ]:


# Add an X & Y Label
plt.xlabel('Days')
plt.ylabel('Days Squared')


# In[ ]:


# If not in Jupyter Notebook use
# plt.show()


# ### Print Multiple Plots

# In[ ]:


# You can print multiple plots at once
# Define the row and column to print the plot with a number assigned to the plot


# In[ ]:


# First subplot with row=1, column=2, plot=1
plt.subplot(1,2,1)


# In[ ]:


# Make the line red
plt.plot(x_1,y_1,'r')


# In[ ]:


# Second subplot with row=1, column=2, plot=2
plt.subplot(1,2,2)


# In[ ]:


# Make the line blue
plt.plot(y_1,x_1,'b')


# ### Using Figure Objects

# In[ ]:


# A figure is an object that contains all the plot elements
# It can contain many axes
# Define width & height in inches
# Dots per inch
fig_1 = plt.figure(figsize=(5,4),dpi=100)


# In[ ]:


# Adds axes with a left, bottom, width and height that ranges from 0 to 1
# which is the percent of the canvas you want to use
axes_1 = fig_1.add_axes([0.1,0.1,0.9,0.9])


# In[ ]:


# Set lables and title
axes_1.set_xlabel('Days')
axes_1.set_ylabel('Days Squared')
axes_1.set_title('Days Squared Chart')


# In[ ]:


# Plot on the axes (If you want a label associated with the legend add it with label)
axes_1.plot(x_1,y_1,label='x / x²')


# In[ ]:


# You can plot to plots using the same axes
axes_1.plot(y_1,x_1,label='x² / x')


# In[ ]:


# Add the optional legend with a location number (best: 0, 
# upper right: 1, upper left: 2, lower left: 3, lower right: 4,
# https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.legend.html)
# or supply a tuple of x & y from lower left
axes_1.legend(loc=0)


# In[ ]:


# You can create axis inside of others
axes_2 = fig_1.add_axes([0.45,0.45,0.4,0.3])
axes_2.set_xlabel('Days')
axes_2.set_ylabel('Days Squared')
axes_2.set_title('Small Days Squared Chart')
axes_2.plot(x_1,y_1,'r')


# In[ ]:


# Add text to plot from central point of 0,0
axes_2.text(0, 40, 'Message')


# ### SubPlots

# In[ ]:


# You can define multiple plots with subplots and it handles creating the axes objects
# axes_2 is a list of axes objects
fig_2, axes_2 = plt.subplots(figsize=(8,4), nrows=1, ncols=3)


# In[ ]:


# Put space between plots
plt.tight_layout()


# In[ ]:


# You can access the plots by index
axes_2[1].set_title('Plot 2')
axes_2[1].set_xlabel('X')
axes_2[1].set_ylabel('X Squared')
axes_2[1].plot(x_1,y_1)


# ### Appearance Options

# In[ ]:


fig_3 = plt.figure(figsize=(6,4))


# In[ ]:


axes_3 = fig_3.add_axes([0,0,1,1])


# In[ ]:


# Default colors (b: blue, g: green, r: red, c: cyan, m: magenta,
# y: yellow, k: black, w: white)
# color="0.75" creates a 75% gray
# You can use hexcodes color="#eeefff"
# You can use color names found next like this color="burlywood"
# https://en.wikipedia.org/wiki/Web_colors
# alpha defines the percentage of opacity


# In[ ]:


# The default line width is 1, so to double it put in 2 and so forth


# In[ ]:


# There are many line styles 
# matplotlib.org/3.1.0/gallery/lines_bars_and_markers/linestyles.html
# You can also provide a sample like '-.'


# In[ ]:


# Markers can mark your provided points on the graph
# https://matplotlib.org/3.3.0/api/markers_api.html
# You can change the markersize as well


# In[ ]:


# markerfacecolor changes the marker fill color
# markeredgecolor changes the marker stroke color
# markeredgewidth changes the markers stroke size


# In[ ]:


axes_3.plot(x_1,y_1,color='navy', alpha=.75, lw=2, ls='-.',
           marker='o', markersize=7, markerfacecolor='y',
           markeredgecolor='y', markeredgewidth=4)


# In[ ]:


# Set the lower and upper bound of x & y axis
axes_3.set_xlim([0,3])
axes_3.set_ylim([0,8])


# In[ ]:


# Add a grid, color, dashes(5pts 1 pt dashes separated by 2pt space)
axes_3.grid(True, color='0.6', dashes=(5, 2, 1, 2))


# In[ ]:


# Set grid background color
axes_3.set_facecolor('#FAEBD7')


# ### Save a Visualization to a File

# In[ ]:


# You can save your plots to numerous file types : png, pdf, ps, eps, svg, pgf, 
fig_3.savefig('1st_plot.png')


# ### Working with Pandas DataFrame

# In[ ]:


# Read in ice cream sales data
ics_df = pd.read_csv('icecreamsales.csv')
ics_df = ics_df.sort_values(by='Temperature')


# In[ ]:


# Convert from Pandas to NumPy array
np_arr = ics_df.values


# In[ ]:


# Get x & y values and put in array
x_2 = np_arr[:,0]
y_2 = np_arr[:,1]


# In[ ]:


fig_4 = plt.figure(figsize=(6,4))
axes_4 = fig_4.add_axes([0,0,1,1])
axes_4.set_title('Ice Cream Sales vs. Temperature')
axes_4.set_xlabel('Temperature')
axes_4.set_ylabel('Ice Cream Sales')
axes_4.plot(x_2,y_2)


# In[ ]:


# Add Annotations by supplying the x & y to point at and the position of the text
# based off of lower left had corner being 0,0
axes_4.annotate('Good Month', xy=(83, 536), xytext=(60, 520), arrowprops=dict(facecolor='black', shrink=0.05),)


# In[ ]:


# Add bars to the plot
plt.bar(x_2,y_2)


# ### TeX Markup

# In[ ]:


# You can use a subset of TeX markup by placing text between $
# matplotlib.org/tutorials/text/mathtext.html
fig_5 = plt.figure(figsize=(5,4),dpi=100)
axes_5 = fig_5.add_axes([0.1,0.1,0.9,0.9])


# In[ ]:


# All listed plus kappa, iota, zeta, nu, rho, eta, xi, omicron, gamma, tau, phi, 
# chi, psi, delta (Capitalize the first letter for uppercase)
axes_5.text(0, 23,  r'$\alpha \beta \sigma \omega \epsilon \mu \pi \theta \lambda$', fontsize=18)


# In[ ]:


# Subscripts, multiletter superscript, sum, fractions, binomial
axes_5.text(0, 18, r'$\delta_i \gamma^{ij} \sum_{i=0}^\infty x_i \frac{3}{4} \binom{3}{4}$', fontsize=18)


# In[ ]:


# Another fraction, sqrt, cbrt, trig functions : 
axes_5.text(0, 13, 
            r'$\frac{8 - \frac{x}{5}}{8} \sqrt{9} \sin(\pi) \sqrt[3]{8} \acute a \div$', 
            fontsize=18)
axes_5.text(0, 8, 
            r'$\bar a \hat a \tilde a \vec a \overline {a} \lim_{x \to 2} f(x) = 5$', 
            fontsize=18)
axes_5.text(0, 3, 
            r'$\geq \leq \ne$', 
            fontsize=18)
axes_5.plot(x_1,y_1)


# ### Histograms

# In[ ]:


# Roll 2 6 sided dies get the sum and plot the histogram
arr_1 = np.random.randint(1,7,5000)
arr_2 = np.random.randint(1,7,5000)
arr_3 = arr_1 + arr_2


# In[ ]:


# Bins reprsent the number of options available 2 thru 12 = 11
# Density returns the frequency of each bin
# Range gets tuple with bin range interested in
# cumulative=True use a cumulative distribution
# histtype='step' genrates a line plot
# orientation='horizontal' 
# color='orange' change bar color
plt.hist(arr_3, bins=11, density=True, stacked=True)


# ### Bar Charts

# In[ ]:


# Analyze where France gets its electricity from
x = ['Nuclear', 'Hydro', 'Coal', 'Gas', 'Solar', 'Wind', 'Other']
per_1 = [71, 10, 3, 7, 2, 4, 3]


# In[ ]:


# Chart variance in usage
variance = [8, 3, 1, 3, 1, 2, 1]


# In[ ]:


# barh makes horizontal chart
# Also yerr, change error color with ecolor
# plt.bar(x, per_1, color='purple', yerr=variance)


# In[ ]:


# Show percentages of males & females in engineering
m_eng = (76, 85, 86, 88, 93)
f_eng = (24, 15, 14, 12, 7)


# In[ ]:


# Get evenly spaced values for each interval
spc = np.arange(5)


# In[ ]:


# Plot bars for men & women
# Can also add yerr, xerr, 
# plt.bar(spc, m_eng, width=0.45, label='Male', edgecolor='k')
# plt.bar(spc + 0.45, f_eng, width=0.45, label='Female', edgecolor='k')


# In[ ]:


# Define x tick names and place in middle of bars
# plt.xticks(spc + 0.45 / 2, ('Aero', 'Chem', 'Civil', 'Elec', 'Mech'))


# In[ ]:


# Plot teachers by sex
t_type = ['Kind', 'Elem', 'Sec', 'Spec']
m_teach = np.array([2, 20, 44, 14])
f_teach = np.array([98, 80, 56, 86])
ind = [x for x, _ in enumerate(t_type)]


# In[ ]:


# Plot stacked bars for men and then women under
plt.bar(ind, m_teach, width=0.45, label='Male', bottom=f_teach)
plt.bar(ind, f_teach, width=0.45, label='Female')

plt.legend(loc='lower right')


# ### Pie Charts

# In[ ]:


import random


# In[ ]:


fig_6 = plt.figure(figsize=(8,5),dpi=100)
axes_6 = fig_6.add_axes([0.1,0.1,0.9,0.9])


# In[ ]:


# Create a pie chart of the number of Pokemon by type
types = ['Water', 'Normal', 'Flying', 'Grass', 'Psychic', 'Bug', 'Fire', 'Poison',
'Ground', 'Rock', 'Fighting', 'Dark', 'Steel', 'Electric', 'Dragon', 'Fairy','Ghost', 'Ice']
poke_num = [133, 109, 101, 98, 85, 77, 68, 66, 65, 60, 57, 54, 53, 51, 50, 50, 46, 40]


# In[ ]:


# Generate a random color array (Use lower values to make darkb)
colors = []
for i in range(18):
    rgb = (random.uniform(0, .5), random.uniform(0, .5), random.uniform(0, .5))
    colors.append(rgb)


# In[ ]:


# Explode biggest 1st slice
explode = [0] * 18
explode[0] = 0.2


# In[ ]:


# Provide values, what to explode and by how much, labels, colors, pct for values,
# whether to shadow, amount to rotate pie, pie text color
wedges, texts, autotexts = plt.pie(poke_num, explode=explode, labels=types, colors=colors, 
                                   autopct='%1.0f%%', shadow=True, startangle=140, 
                                   textprops=dict(color="w"))


# In[ ]:


# Create legend to right and move off pie with 1-1.5 axes point width
plt.legend(wedges, types, loc='right', bbox_to_anchor=(1, 0, 0.5, 1))


# ### Plot Timeseries

# In[ ]:


import datetime


# In[ ]:


# I'll show other ways of doing this when I cover Matplotlib Finance
# Get Yahoo GOOG csv file and convert to NumPy array
# https://finance.yahoo.com/quote/GOOG/history/
goog_data = pd.read_csv('GOOG.csv')
goog_data_np = goog_data.to_numpy()


# In[ ]:


# Get array of prices in 5th column
goog_cp = goog_data_np[:,4]
goog_cp


# In[ ]:


# Get NumPy array with just weekdays between dates excluding holidays
holidays = [datetime.datetime(2020,5,25), datetime.datetime(2020,8,19)]
date_arr = pd.bdate_range(start='5/20/2020', end='8/19/2020', freq='C', holidays=holidays)
date_arr_np = date_arr.to_numpy()


# In[ ]:


fig_7 = plt.figure(figsize=(8,5),dpi=100)
axes_7 = fig_7.add_axes([0.1,0.1,0.9,0.9])


# In[ ]:


plt.plot(date_arr_np, goog_cp)


# ### Tables

# In[ ]:


# Format column data to 2 decimals
goog_data['Open'] = pd.Series([round(val, 2) for val in goog_data['Open']], index = goog_data.index)

goog_data['High'] = pd.Series([round(val, 2) for val in goog_data['High']], index = goog_data.index)

goog_data['Low'] = pd.Series([round(val, 2) for val in goog_data['Low']], index = goog_data.index)

goog_data['Close'] = pd.Series([round(val, 2) for val in goog_data['Close']], index = goog_data.index)

goog_data['Adj Close'] = pd.Series([round(val, 2) for val in goog_data['Adj Close']], index = goog_data.index)


# In[ ]:


# Get most recent last 5 days of stock data
stk_data = goog_data[-5:]
stk_data


# In[ ]:


# Define headers
col_head = ('Date','Open','High','Low','Close','Adj Close','Volume')


# In[ ]:


stk_data_np = stk_data.to_numpy()
stk_data_np


# In[ ]:


# Add padding around cells in table
plt.figure(linewidth=2, tight_layout={'pad':.5}, figsize=(5,3))


# In[ ]:


# Get rid of axes and plot box
axes_8 = plt.gca()
axes_8.get_xaxis().set_visible(False)
axes_8.get_yaxis().set_visible(False)
plt.box(on=None)


# In[ ]:


# np.full returns an array filled with 0.1
# cm is a colormap object we are using to use a default blue color
# matplotlib.org/3.1.0/tutorials/colors/colormaps.html
ccolors = plt.cm.Blues(np.full(len(col_head), 0.2))


# In[ ]:


# Receives data, loc, list of column headers, column header color as array of colors
# You can also add rowLabel, rowColours, rowLoc: Text alignment
the_table = plt.table(cellText=stk_data_np, loc='center', colLabels=col_head, colColours=ccolors)


# In[ ]:


# Set table font size
the_table.set_fontsize(14)
the_table.scale(3, 2.5)


# ### Scatterplots

# In[ ]:


# Country array
cnt_arr = np.array(['Australia','Brazil','Canada','Chile','France','Germany','Greece',
                   'Iceland','India','Iran','Italy','Mexico','New Zealand','Nigeria',
                   'Norway','Pakistan','Peru','Russia','Saudi Arabia','Singapore',
                   'South Africa','Spain','Sweden','Turkey','UK','US'])


# In[ ]:


# Death rate per 100k Coronavirus
dr_arr = np.array([1.8,53,24.5,56.5,45.4,11.2,2.2,
                   2.8,4,24.6,58.6,46.3,.5,.5,
                   4.9,2.9,83.3,11,10.4,.5,
                   21.5,61.6,56.9,7.3,62.4,52.9])


# In[ ]:


# Daily confirmed cases (Tests)
test_arr = np.array([110,7197,600,1862,1636,1103,35,
                   10,295,1658,1226,2490,8,243,
                   48,1395,1101,4447,1443,280,
                   2830,1602,447,1205,1546,24988])


# In[ ]:


# Dot size Confirmed cases
cc_arr = np.array([24236,3456652,125408,390037,256534,229706,7684,
                   2035,2836925,350279,255278,537031,1654,50488,
                   10162,290445,549321,935066,302686,56031,
                   596060,370867,85411,253108,323008,5529824])


# In[ ]:


cc_arr_sm = cc_arr / 1000
color_arr = np.random.rand(26)


# In[ ]:


plt.title('Death Rate per 100k vs. Confirmed Cases')
plt.xlabel('Death Rate per 100k')
plt.ylabel('Confirmed Cases')
plt.scatter(dr_arr,test_arr,s=cc_arr_sm,c=color_arr,alpha=0.5)


# ### 3D Surface

# In[ ]:


# Needed for creating 3D axes
from mpl_toolkits import mplot3d


# In[ ]:


fig_9 = plt.figure(figsize=(8,5),dpi=100)
axes_9 = fig_9.add_axes([0.1,0.1,0.9,0.9], projection='3d')


# In[ ]:


# Create a 3D scatterplot
# Te darker points are represented that way to seem closer to you
z_3 = 15 * np.random.random(100)
x_3 = np.sin(z_3) * np.random.randn(100)
y_3 = np.cos(z_3) * np.random.randn(100)
# axes_9.scatter3D(x_3, y_3, z_3, c=z_3, cmap='Blues')


# In[ ]:


# You can create contour plots by defining a function for z based on x & y
def get_z(x, y):
    return np.sin(np.sqrt(x**2 + y**2))

x_4 = np.linspace(-6, 6, 30)
y_4 = np.linspace(-6, 6, 30)


# In[ ]:


# Creates a rectangular grid out of 2 given 1D arrays
x_4, y_4 = np.meshgrid(x_4, y_4)
z_4 = get_z(x_4,y_4)


# In[ ]:


# Change viewing angle to reorient camera 60 degrees and rotate 55
axes_9.view_init(45,55)


# In[ ]:


# Provide x, y, z, contours and color map
# axes_9.contour3D(x_4,y_4,z_4,80,cmap='Blues')
axes_9.set_xlabel('x')
axes_9.set_ylabel('y')
axes_9.set_zlabel('z')


# In[ ]:


# You can create wireframes
# axes_9.plot_wireframe(x_4,y_4,z_4,color='blue')


# In[ ]:


# You can create surface plots which is wireframe with filled faces
axes_9.plot_surface(x_4,y_4,z_4, rstride=1, cstride=1,cmap='Blues',edgecolor='none')


# ### Matplotlib Finance

# In[ ]:


# We need this module to handle the calculations
import mplfinance as mpf


# In[ ]:


# Get stock data as DataFrame and define index
goog_df = pd.read_csv('GOOG.csv',index_col=0,parse_dates=True)
goog_df.index.name = 'Date'

goog_df.shape


# In[ ]:


# A candlestick chart demonstrates the daily open, high, low and closing price of a stock
# mpf.plot(goog_df,type='candle')


# In[ ]:


# Plot price changes
# mpf.plot(goog_df,type='line')


# In[ ]:


# Moving averages provide trend information (Average of previous 4 observations)
# mpf.plot(goog_df,type='ohlc',mav=4)


# In[ ]:


# You can plot multiple MAVs, volume, non-trading days
mpf.plot(goog_df,type='ohlc',mav=(3,5,7),volume=True,show_nontrading=True)


# In[ ]:


# You can make additonal charts with intraday data


# ### Heatmaps

# In[ ]:


# A heatmap is a color coded representation of data from a 2D list
symptoms = ["Coronavirus","Influenza","Pneumonia","Dyspnea"]
dates = ["Jun28","Jul05","Jul12","Jul19","Jul26","Aug02","Aug09","Aug16","Aug21"]
symp_per = np.array([[5.2, 5.5, 5.7, 5.6, 5.3, 5.1, 5.0, 4.9, 5.3],
                    [3.5, 4.0, 4.3, 3.9, 3.5, 3.2, 2.7, 2.2, 2.0],
                    [1.8, 2.2, 2.3, 2.2, 2.1, 1.9, 1.7, 1.4, 1.3],
                    [1.0, 1.1, 1.1, 1.0, 0.9, 0.8, 0.8, 0.8, 0.7]])

fig_10, axes_10 = plt.subplots()


# In[ ]:


# Dfine data to use and color map
im = axes_10.imshow(symp_per, cmap="Wistia")


# In[ ]:


# Add ticks at data points and labels
axes_10.set_xticks(np.arange(len(dates)))
axes_10.set_yticks(np.arange(len(symptoms)))
axes_10.set_xticklabels(dates)
axes_10.set_yticklabels(symptoms)


# In[ ]:


# Rotate labels on the bottom so they don't overlap
plt.setp(axes_10.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")


# In[ ]:


# Loop over data dimensions and create text annotations.
for i in range(len(symptoms)):
    for j in range(len(dates)):
        text = axes_10.text(j, i, symp_per[i, j], ha="center", va="center", color="k",fontweight="bold")

