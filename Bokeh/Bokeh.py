#!/usr/bin/env python
# coding: utf-8

# # Bokeh Tutorial

# ### Bokeh is an interactive visualization library.
# 
# Currently, Bokeh has two main interfaces:
# * bokeh.models - A low-level interface that provides the most flexibility to application developers
# * bokeh.plotting - A higher-level interface centered around composing visual glyphs
# 
# Bokeh can also be used with the HoloViews library.
# 
# bokeh.charts is deprecated.

# ### Installation
# 
# Go to python package index at python.org and see install instructions for bokeh.

# ### Documentation

# In[ ]:


from IPython.display import IFrame

documentation = IFrame(src='https://bokeh.pydata.org/en/latest/', width=1000, height=450)
display(documentation)


# ### Imports
# Some imports can be imported in different ways and appear to do the same thing (i.e.-bokeh.io.show, bokeh.plotting.show)

# In[ ]:


# standard bokeh imports
from bokeh.io import output_notebook, show, reset_output


# In[ ]:


# other bokeh imports
import bokeh
from bokeh.plotting import figure


# In[ ]:


# other imports
import numpy as np
import pandas as pd
from vega_datasets import data as vds


# ### Troubleshooting
# 
# * reset_output(), then output_notebook() to keep from opening new tabs and display plots in notebook
# * from bokeh.charts import 'Plot Type' is deprecated, use from bokeh.plotting import figure instead
# * if something is not working, try to update to the latest version of bokeh

# ### Sample Data Sets

# In[ ]:


from bokeh.sampledata import iris

# sample data set (dataframe)
iris_dataset = iris.flowers
iris_dataset.head()


# ### To Display Plots In Notebook

# In[ ]:


# configure the default output state to generate output in notebook cells when show() is called
# in colab, output_notebook() is called in each cell (this is not always the case)
output_notebook()


# In[ ]:


# output_notebook?


# ### To Save Plots

# In[ ]:


# if not using a notebook environment, use this to save/open your bokeh plots
# save your plot to html file or click save icon next to plot
# output_file()

'''
comment out output_file(), run reset_output(), then run output_notebook() to keep from opening new tabs 
and display plots in notebook
'''


# ### Steps to Create Plots
# 
# 1. create figure - used to create/house plot
# 
# 2. call plot (glyph) method (types = line, bar, scatter, etc.)
# 
# 3. show figure plot

# ### Column Data Source
# 
# The ColumnDataSource is a data source used throughout Bokeh. Bokeh often creates the ColumnDataSource automatically, however there are times when it is useful to create them explicitly.
# 
# The ColumnDataSource is a (dictionary) mapping of column names (strings) to sequences of values. The mapping is provided by passing a dictionary with string keys and lists (or similar data structures) as values.

# In[ ]:


from bokeh.models import ColumnDataSource

column_data_source = ColumnDataSource({'A': [1, 2, 3, 4, 5],
                                       'B': [5, 4, 3, 2, 1],
                                       'C': [1, 3, 5, 1, 2]})

column_data_source.data


# ### Line Plot

# In[ ]:


from bokeh.models import HoverTool


# In[ ]:


# data
x_line = np.arange(10)
y_line = np.random.rand(10)


# In[ ]:


# line plot
line_plot = figure(plot_width=500, plot_height=325, title='Line Plot', x_axis_label='x', y_axis_label='y')
line_plot.line(x_line, y_line, legend='line', line_width=2)


# In[ ]:


# add hover tool
line_plot.add_tools(HoverTool())

# another way to set axis labels
# line_plot.xaxis.axis_label = 'x-axis'
# line_plot.yaxis.axis_label = 'y-axis'


# In[ ]:


show(line_plot)


# Plot Tools = Documentation, Pan, Box Zoom, Wheel Zoom, Save, Reset, Hover, Bokeh Plot Tools Information, Additional Tools Can Be Added

# ### Line Plot ( Multiple Lines )

# In[ ]:


# data
multi_line_x = np.arange(10)
multi_line_y1 = np.random.rand(10)
multi_line_y2 = np.random.rand(10)
multi_line_y3 = np.random.rand(10)


# In[ ]:


# plot 
multi_line_plot = figure(plot_width=500, plot_height=300, toolbar_location='below')
multi_line_plot.line(multi_line_x, multi_line_y1, color='red', line_width=3)
multi_line_plot.line(multi_line_x, multi_line_y2, color='blue', line_width=3)
multi_line_plot.line(multi_line_x, multi_line_y3, color='yellow', line_width=3)
show(multi_line_plot)


# ### Bar Charts

# In[ ]:


# data
x_bar = ['category1', 'category2', 'category3', 'category4', 'category5']
y_bar = np.random.rand(5)*10


# In[ ]:


# sort data (sort x by its cooresponding y)
sorted_categories = sorted(x_bar, key=lambda x: y_bar[x_bar.index(x)], reverse=True)


# In[ ]:


# plot
bar_chart = figure(x_range=sorted_categories, title='Bar Plot', x_axis_label='x', y_axis_label='y', plot_height=300)
bar_chart.vbar(x_bar, top=y_bar, color='blue', width=0.5)
bar_chart.y_range.start = 0
show(bar_chart)

# use hbar for horizontal bar chart


# ### Stacked Bar Chart

# In[ ]:


# note - appears to work fine with dataframe or column data source converted dataframe
# if dataframe does not work, use column data source

stacked_bar_df = pd.DataFrame({'y': [1, 2, 3, 4, 5],
                               'x1': [1, 2, 4, 3, 4],
                               'x2': [1, 4, 2, 2, 3]})


# In[ ]:


cds_stacked_bar_df = ColumnDataSource(stacked_bar_df)


# In[ ]:


stacked_bar_chart = figure(plot_width=600, plot_height=300, title='stacked bar chart')


# In[ ]:


stacked_bar_chart.hbar_stack(['x1', 'x2'], 
                             y='y', 
                             height=0.8, 
                             color=('grey', 'lightgrey'), 
                             source=cds_stacked_bar_df)


# In[ ]:


show(stacked_bar_chart)


# ### Grouped Bar Chart

# In[ ]:


from bokeh.core.properties import value
from bokeh.transform import dodge


# In[ ]:


# data
categories = ['category1', 'category2', 'category3']
grouped_bar_df = pd.DataFrame({'categories' : categories,
                               '2015': [2, 1, 4],
                               '2016': [5, 3, 3],
                               '2017': [3, 2, 4]})


# In[ ]:


# plot
grouped_bar = figure(x_range=categories, y_range=(0, 10), plot_height=250)


# In[ ]:


# offsets bars / bar locations on axis
dodge1 = dodge('categories', -0.25, range=grouped_bar.x_range)
dodge2 = dodge('categories',  0.0,  range=grouped_bar.x_range)
dodge3 = dodge('categories',  0.25, range=grouped_bar.x_range)

grouped_bar.vbar(x=dodge1, top='2015', width=0.2, source=grouped_bar_df, color='gray', legend=value('2015'))
grouped_bar.vbar(x=dodge2, top='2016', width=0.2, source=grouped_bar_df, color='blue', legend=value('2016'))
grouped_bar.vbar(x=dodge3, top='2017', width=0.2, source=grouped_bar_df, color='green', legend=value('2017'))


# In[ ]:


# format legend
grouped_bar.legend.location = 'top_left'
grouped_bar.legend.orientation = 'horizontal'


# In[ ]:


show(grouped_bar)


# ### Stacked Area Chart

# In[ ]:


stacked_area_df = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                                'y1': [1, 2, 4, 3, 4],
                                'y2': [1, 4, 2, 2, 3]})


# In[ ]:


stacked_area_plot = figure(plot_width=600, plot_height=300)


# In[ ]:


stacked_area_plot.varea_stack(['y1', 'y2'],
                              x='x',
                              color=('green', 'lightgreen'),
                              source=stacked_area_df)


# In[ ]:


show(stacked_area_plot)


# In[ ]:


# stacked_area_plot.varea_stack?


# ### Scatter Plots

# In[ ]:


# vega data sets cars data
cars = vds.cars()
cars.tail()


# In[ ]:


# data
x_scatter = cars.Weight_in_lbs
y_scatter = cars.Miles_per_Gallon


# In[ ]:


# plot 
scatter_plot = figure(plot_width=500, plot_height=300, x_axis_label='Weight_in_lbs', y_axis_label='Miles_per_Gallon')
scatter_plot.circle(x_scatter, y_scatter, size=15, line_color='navy', fill_color='orange', fill_alpha=0.5)
show(scatter_plot)


# Other scatter plot variations include: cross, x, diamond, diamond_cross, circle_x, circle_cross, triangle, 
# inverted_triangle, square, square_x, square_cross, asterisk

# In[ ]:


# vega data sets iris data
iris = vds.iris()
iris.tail()


# ### Scatter Plot Subgroups

# In[ ]:


# scatter plot subgroups using iris data

from bokeh.transform import factor_cmap, factor_mark


# In[ ]:


# data
# use vega_datasets iris data

# plot 
species = ['setosa', 'versicolor', 'virginica']
markers = ['hex', 'cross', 'triangle']


# In[ ]:


scatter_plot_subgroups = figure(plot_width=600, 
                                plot_height=400, 
                                title ='Iris', 
                                x_axis_label='petalLength', 
                                y_axis_label='petalWidth')


# In[ ]:


scatter_plot_subgroups.scatter(x='petalLength',
                               y='petalWidth',
                               source=iris,
                               legend='species',
                               fill_alpha=0.5,
                               size=15,
                               color=factor_cmap(field_name='species', palette='Dark2_3', factors=species),
                               marker=factor_mark('species', markers, species)
                              )


# In[ ]:


# move legend
scatter_plot_subgroups.legend.location = 'top_left'
show(scatter_plot_subgroups)


# In[ ]:


# scatter_plot_subgroups.scatter?


# ### Subplots

# In[ ]:


from bokeh.layouts import gridplot


# In[ ]:


# data
subplot_x1 = cars['Acceleration']; subplot_y1 = cars['Miles_per_Gallon']
subplot_x2 = cars['Cylinders']; subplot_y2 = cars['Miles_per_Gallon']
subplot_x3 = cars['Horsepower']; subplot_y3 = cars['Miles_per_Gallon']
subplot_x4 = cars['Weight_in_lbs']; subplot_y4 = cars['Miles_per_Gallon']


# In[ ]:


# figures
subplot1 = figure(plot_width=300, plot_height=300)
subplot2 = figure(plot_width=300, plot_height=300)
subplot3 = figure(plot_width=300, plot_height=300)
subplot4 = figure(plot_width=300, plot_height=300)


# In[ ]:


# plots
subplot1.circle(subplot_x1, subplot_y1)
subplot2.circle(subplot_x2, subplot_y2)
subplot3.circle(subplot_x3, subplot_y3)
subplot4.circle(subplot_x4, subplot_y4)


# In[ ]:


# subplots gridplot
grid = gridplot([subplot1, subplot2, subplot3, subplot4], ncols=2)


# In[ ]:


# show
show(grid)


# ### Link Plots

# In[ ]:


from bokeh.layouts import gridplot

linked_data_x = np.arange(10)
linked_data_y = np.random.rand(10)


# In[ ]:


# linked plot 1
linked_plot1 = figure(width=250, height=250)
linked_plot1.circle(linked_data_x, linked_data_y)


# In[ ]:


# create new plots and share both ranges
linked_plot2 = figure(width=250, height=250, x_range=linked_plot1.x_range, y_range=linked_plot1.y_range)
linked_plot2.line(linked_data_x, linked_data_y)

linked_plot3 = figure(width=250, height=250, x_range=linked_plot1.x_range, y_range=linked_plot1.y_range)
linked_plot3.vbar(linked_data_x, top=linked_data_y, width=0.5)


# In[ ]:


# the subplots in a gridplot
linked_gridplot = gridplot([[linked_plot1, linked_plot2, linked_plot3]])


# In[ ]:


# show the results
show(linked_gridplot)


# ### Linked Selection - Box Select, Lasso Select

# In[ ]:


# data
seattle_weather = vds.seattle_weather()
seattle_weather.tail()


# In[ ]:


from bokeh.transform import factor_cmap, factor_mark

TOOLS = 'box_select, lasso_select, reset, wheel_zoom, pan'

weather_types = ['drizzle', 'rain', 'sun', 'snow', 'fog']
weather_markers = ['hex', 'cross', 'triangle', 'square', 'circle_x']


# In[ ]:


# use ColumnDataSource for linking interactions
seattle_weather_source = ColumnDataSource(seattle_weather)


# In[ ]:


# scatter plot 1
weather_scatter = figure(plot_width=900, plot_height=300, y_axis_label='Temp', x_axis_type='datetime', tools=TOOLS)
weather_scatter.circle('date', 'temp_max', size=15, fill_alpha=0.1, source=seattle_weather_source)


# In[ ]:


# scatter plot 2
weather_scatter_zoom = figure(plot_width=900, plot_height=500, x_axis_type='datetime', tools=TOOLS)
weather_scatter_zoom.scatter('date', 
                             'temp_max', 
                             size=15, 
                             fill_alpha=0.1,
                             color=factor_cmap(field_name='weather', palette='Dark2_5', factors=weather_types),
                             marker=factor_mark('weather', weather_markers, weather_types),
                             legend='weather',
                             source=seattle_weather_source
                             )


# In[ ]:


# shared data between plots helps the linked selection to work

# format legend
weather_scatter_zoom.legend.location = 'top_left'
weather_scatter_zoom.legend.orientation = 'horizontal'


# In[ ]:


weather_grid = gridplot([[weather_scatter], [weather_scatter_zoom]])
show(weather_grid)


# ### Labels and Annotations

# In[ ]:


from bokeh.models.annotations import Label, LabelSet, Arrow
from bokeh.models.arrow_heads import NormalHead

output_notebook()


# In[ ]:


# data
fig_with_label_data = ColumnDataSource({'x': np.arange(10), 
                                        'y': [4, 7, 5, 5, 9, 2, 3, 4, 3, 4]})


# In[ ]:


# plot
fig_with_label = figure()
fig_with_label.line(x='x', y='y', source=fig_with_label_data)


# In[ ]:


# add label
label = Label(x=4, y=9, x_offset=10, text='Higest Point', text_baseline='middle')
fig_with_label.add_layout(label)


# In[ ]:


# add multiple labels
labels = LabelSet(x='x', y='y', text='y', level='glyph', source=fig_with_label_data)
fig_with_label.add_layout(labels)


# In[ ]:


# arrow annotation
fig_with_label.add_layout(Arrow(end=NormalHead(fill_color='orange'), x_start=5, y_start=7.5, x_end=4.5, y_end=8.8))


# In[ ]:


show(fig_with_label)


# ### Color Bar

# In[ ]:


cars.head()


# In[ ]:


from bokeh.models import LinearColorMapper, ColorBar
from bokeh.transform import transform


# In[ ]:


# data
# use vega_datasets cars data

# map numbers in a range low, high - linearly into a sequence of colors (a palette)
color_mapper = LinearColorMapper(palette='Viridis256', low=cars.Weight_in_lbs.min(), high=cars.Weight_in_lbs.max())


# In[ ]:


# plot
colorbar_fig = figure(plot_width=600, plot_height=400, x_axis_label='Horsepower', y_axis_label='Miles_per_Gallon')
colorbar_fig.circle(x='Horsepower', 
                    y='Miles_per_Gallon',
                    source=cars,
                    color=transform('Weight_in_lbs', color_mapper), 
                    size=15, 
                    alpha=0.5)


# In[ ]:


# render a color bar based on a color mapper
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, location=(0,0), title='Weight')
colorbar_fig.add_layout(color_bar, 'right')


# In[ ]:


show(colorbar_fig)


# ### Map

# In[ ]:


# convert longitude, latitude to mercator coordinates
# example - Detroit Michigan 42.334197, -83.047752


# In[ ]:


from pyproj import Proj, transform


# In[ ]:


def create_coordinates(long_arg,lat_arg):
    in_wgs = Proj(init='epsg:4326')
    out_mercator = Proj(init='epsg:3857')
    long, lat = long_arg, lat_arg
    mercator_x, mercator_y = transform(in_wgs, out_mercator, long, lat)
    print(mercator_x, mercator_y)


# In[ ]:


# Detroit
create_coordinates(-83.047752,42.334197)


# In[ ]:


# Cleveland
create_coordinates(-81.694703,41.499437)


# In[ ]:


# Chicago 
create_coordinates(-87.629849,41.878111)


# In[ ]:


from bokeh.tile_providers import get_provider, Vendors


# In[ ]:


tile_provider = get_provider(Vendors.CARTODBPOSITRON)
# tile_provider = get_provider(Vendors.STAMEN_TONER_BACKGROUND)


# In[ ]:


# range bounds supplied in web mercator coordinates
m = figure(plot_width=800, 
           plot_height=400,
           x_range=(-12000000, 9000000), 
           y_range=(-1000000, 7000000), 
           x_axis_type='mercator', 
           y_axis_type='mercator')


# In[ ]:


m.add_tile(tile_provider)


# In[ ]:


m.circle(x=-9244833, y=5211172, size=10, color='red')
m.circle(x=-9094212, y=5086289, size=10, color='blue')
m.circle(x=-9754910, y=5142738, size=10, color='orange')


# In[ ]:


show(m)


# ### Interactive Widgets

# In[ ]:


# change size of scatter plot circles
from bokeh.layouts import column
from bokeh.models import Slider


# In[ ]:


# create figure and plot
change_plot_size = figure(plot_width=600, plot_height=300)
change_plot_size_r = change_plot_size.circle([1,2,3,4,5], [3,2,5,6,4], radius=0.1, alpha=0.5)


# In[ ]:


# create widget and link
slider = Slider(start=0.1, end=1, step=0.01, value=0.2)
slider.js_link('value', change_plot_size_r.glyph, 'radius')


# In[ ]:


show(column(change_plot_size, slider))


# In[ ]:


from sklearn import linear_model
from bokeh.layouts import layout
from bokeh.models import Toggle
import numpy as np


# In[ ]:


# data
x = [1,2,3,4,5,6,7,8,9,10]
X = np.array(x).reshape(-1, 1)
y = [2,2,4,1,5,6,8,2,3,7]
Y = np.array(y).reshape(-1, 1)


# In[ ]:


# linear regression object
regr = linear_model.LinearRegression()


# In[ ]:


# fit linear model
regr.fit(X, Y)


# In[ ]:


# make predictions
pred = regr.predict(X)


# In[ ]:


# plot with regression line
regr_plot = figure(plot_width=500, plot_height=300)
regr_plot.scatter(x, y, size=10)
regr_line = regr_plot.line(x, pred.flatten(), line_color='red')

toggle_button = Toggle(label='line of best fit', button_type='success', active=True)
toggle_button.js_link('active', regr_line, 'visible')


# In[ ]:


show(layout([regr_plot], [toggle_button]))


# In[ ]:


# slider.js_link?


# ### Interactive Widgets with ipywidgets

# In[ ]:


seattle_weather['year'] = pd.DatetimeIndex(seattle_weather['date']).year
seattle_weather.tail()


# In[ ]:


import ipywidgets
from bokeh.io import push_notebook
from bokeh.models import Range1d


# In[ ]:


sw = seattle_weather.copy()


# In[ ]:


# widget
drop_down = ipywidgets.Dropdown(options=[2012, 2013, 2014, 2015],
                                value=2012,
                                description='years:',
                                disabled=False)


# In[ ]:


# data
x_bar_data_ipyw = ['precipitation', 'temp_max', 'temp_min', 'wind']
y_bar_data_ipyw = [sw[sw.year==2012]['precipitation'].mean(), 
                   sw[sw.year==2012]['temp_max'].mean(), 
                   sw[sw.year==2012]['temp_min'].mean(), 
                   sw[sw.year==2012]['wind'].mean()]


# In[ ]:


# figure and plot
bar_chart_interactive = figure(x_range=x_bar_data_ipyw, plot_height=300)
bar_ipyw = bar_chart_interactive.vbar(x_bar_data_ipyw, top=y_bar_data_ipyw, color='green', width=0.5)
bar_chart_interactive.y_range=Range1d(0, 18)


# In[ ]:


# function - bar chart
def weather_averages(year):
    if year == 2012: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2012]['precipitation'].mean(), 
                                            sw[sw.year==2012]['temp_max'].mean(), 
                                            sw[sw.year==2012]['temp_min'].mean(), 
                                            sw[sw.year==2012]['wind'].mean()]
    elif year == 2013: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2013]['precipitation'].mean(), 
                                            sw[sw.year==2013]['temp_max'].mean(), 
                                            sw[sw.year==2013]['temp_min'].mean(), 
                                            sw[sw.year==2013]['wind'].mean()]
    elif year == 2014: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2014]['precipitation'].mean(), 
                                            sw[sw.year==2014]['temp_max'].mean(), 
                                            sw[sw.year==2014]['temp_min'].mean(), 
                                            sw[sw.year==2014]['wind'].mean()]
    elif year == 2015: 
        bar_ipyw.data_source.data['top'] = [sw[sw.year==2015]['precipitation'].mean(), 
                                            sw[sw.year==2015]['temp_max'].mean(), 
                                            sw[sw.year==2015]['temp_min'].mean(), 
                                            sw[sw.year==2015]['wind'].mean()]
    push_notebook()

show(bar_chart_interactive, notebook_handle=True)


# In[ ]:


# interaction
ipywidgets.interact(weather_averages, year=drop_down)


# In[ ]:


sw.groupby('year').mean().T


# ### More
# * Embed images, etc. in tooltip example (Configuring Plot Tools - Mouse over the dots)
# * More examples in documentation
