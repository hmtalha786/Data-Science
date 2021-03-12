#!/usr/bin/env python
# coding: utf-8

# ## Pandas Imports
# 
# import pandas as pd. Simply imports the library the current namespace, but rather than using the name pandas , it's instructed to use the name pd instead. This is just so you can do pd.Same with import numpy as np.

# In[ ]:


import numpy as np
import pandas as pd


# ## Series
# 
# Pandas Series is a one-dimensional labeled array capable of holding data of any type (integer, string, float, python objects, etc.). 
# 
# The axis labels are collectively called index. Pandas Series is nothing but a column in an excel sheet.Labels need not be unique but must be a hashable type. The object supports both integer and label-based indexing and provides a host of methods for performing operations involving the index.

# In[ ]:


# Pandas uses something called a dataframe. It is a 
# 2D data structure that can hold multiple data types.
# Columns have labels.


# In[ ]:


# Series are built on top of NumPy arrays. 
# Create a series by first creating a list
list_1 = ['a', 'b', 'c', 'd']


# In[ ]:


# I can define that I want the series indexes to be the provided labels
labels = [1, 2, 3, 4]
ser_1 = pd.Series(data=list_1, index=labels)


# In[ ]:


# You can also add a NumPy array
arr_1 = np.array([1, 2, 3, 4])
ser_2 = pd.Series(arr_1)


# In[ ]:


# You can quickly add labels and values with a dictionary
dict_1 = {"f_name": "Derek", "l_name": "Banas", "age": 44}
ser_3 = pd.Series(dict_1)


# In[ ]:


# Get data by label
ser_3["f_name"]


# In[ ]:


# You can get the datatype
ser_2.dtype


# In[ ]:


# You can perform math operations on series
ser_2 + ser_2
ser_2 - ser_2
ser_2 * ser_2
ser_2 / ser_2


# In[ ]:


# You can pass them into NumPy methods
# See NumPy tutorial for more math methods
np.exp(ser_2)


# In[ ]:


# The difference between Series and ndarray is that operations
# align by labels
# Create a series from a dictionary
ser_4 = pd.Series({4: 5, 5: 6, 6: 7, 7: 8})


# In[ ]:


# If labels don't align you will get NaN
ser_2 + ser_4


# In[ ]:


# You can assign names to series
ser_4 = pd.Series({8: 9, 9: 10}, name='rand_nums')
ser_4.name


# # DataFrames

# DataFrames are the most commonly used data structure with Pandas. They are made up of multiple series that share the same index / label. They can contain multiple data types. They can be created from dicts, series, lists or other dataframes. 

# ## Creating DataFrames

# In[ ]:


from numpy import random


# In[ ]:


# Create random matrix 2x3 with values between 10 and 50
arr_2 = np.random.randint(10, 50, size=(2, 3))


# In[ ]:


# Create DF with data, row labels & column labels
df_1 = pd.DataFrame(arr_2, ['A', 'B'], ['C', 'D', 'E'])


# In[ ]:


# Create a DF from multiple series in a dict
# If series are of different lengthes extra spaces are NaN
dict_3 = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df_2 = pd.DataFrame(dict_3)
df_2


# In[ ]:


# from_dict accepts a column labels and lists
pd.DataFrame.from_dict(dict([('A', [1,2,3]), ('B', [4,5,6])]))


# In[ ]:


# You can assign the keys as row labels and column labels separate
# with orient='index'
pd.DataFrame.from_dict(dict([('A', [1,2,3]), ('B', [4,5,6])]),orient='index', columns=['one','two','three'])


# In[ ]:


# Get number of rows and columns as tuple
print(df_1.shape)


# ## Editing & Retrieving Data

# In[ ]:


# Grab a column
df_1['C']


# In[ ]:


# Get multiple columns
df_1[['C', 'E']]


# In[ ]:


# Grabb a row as a series
df_1.loc['A']


# In[ ]:


# Grab row by index position
df_1.iloc[1]


# In[ ]:


# Grab cell with Row & Column
df_1.loc['A', 'C']


# In[ ]:


# Grab multiple cells by defining rows wanted & the
# columns from those rows
print(df_1.loc[['A', 'B'], ['D', 'E']])


# In[ ]:


# Make new column
df_1['Total'] = df_1['C'] + df_1['D'] + df_1['E']
df_1


# In[ ]:


# You can perform multiple calculations
df_2['mult'] = df_2['one'] * df_2['two']
df_2


# In[ ]:


# Make a new row by appending
dict_2 = {'C': 44, 'D': 45, 'E': 46}
new_row = pd.Series(dict_2, name='F')
df_1 = df_1.append(new_row)


# In[ ]:


# Delete column and set inplace to True which is required
# because Pandas tries to help you not delete data
# by accident
df_1.drop('Total', axis=1, inplace=True)
df_1


# In[ ]:


# Delete a row
df_1.drop('B', axis=0, inplace=True)
df_1


# In[ ]:


# Create a new column and make it the index
df_1['Sex'] = ['Men', 'Women']
df_1.set_index('Sex', inplace=True)


# In[ ]:


# You can reset index values to numbers
#df_1.reset_index(inplace=True)
df_1


# In[ ]:


# Assign can be used to create a column while leaving the
# original DF untouched
df_2.assign(div=df_2['one'] / df_2['two'])


# In[ ]:


# You can pass in a function as well
df_2.assign(div=lambda x: (x['one'] / x['two']))


# In[ ]:


# Combine DataFrames while keeping df_3 data unless
# there is a NaN value
df_3 = pd.DataFrame({'A': [1., np.nan, 3., np.nan]})
df_4 = pd.DataFrame({'A': [8., 9., 2., 4.]})
df_3.combine_first(df_4)


# ## Conditional Selection

# In[ ]:


arr_2 = np.random.randint(10, 50, size=(2, 3))


# In[ ]:


df_1 = pd.DataFrame(arr_2, ['A', 'B'], ['C', 'D', 'E'])
print(df_1)


# In[ ]:


# You can use conditional operators to retrieve a table
# based on the condition
print("Greater than 40\n", df_1 > 40.0)


# In[ ]:


# You can use comparison operater functions as well like
# gt, lt, ge, le, eq, ne
print("Greater than 45\n", df_1.gt(45.0))


# In[ ]:


# You can place conditions in brackets as well
bool_1 = df_1 >= 45.0
df_1[bool_1]


# In[ ]:


# Get bools for a column
df_1['E'] > 40


# In[ ]:


# Return a row if cell value in column matches a condition
df_1[df_1['E']>30]


# In[ ]:


# You can focus on a column based on resulting dataframe
df_2 = df_1[df_1['E']>30]
df_2['C']


# In[ ]:


# You can stack these commands
print(df_1[df_1['E']>20]['C'])


# In[ ]:


# You can also grab multiple columns
print(df_1[df_1['E']>20][['C', 'D']])


# In[ ]:


# You can use multiple conditions
arr_3 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df_2 = pd.DataFrame(arr_3, ['A', 'B', 'C'], ['X', 'Y', 'Z'])
print(df_2, "\n")


# In[ ]:


# You can use or | to combine conditions as well
df_2[(df_2['X']>3) & (df_2['X']<7)]


# ## File Input / Output 

# Pandas can work with the following types of data : CSV, Plain Text, JSON, XML, PDF, SQL, HTML, XLSX, DOCX, ZIP, Images Hierarchical Data Format, MP3, and MP4.

# In[ ]:


import pymysql


# In[ ]:


# Read a CSV file
# Type pd.read_ [TAB] to see the file types you can read
cs_df = pd.read_csv('ComputerSales.csv')


# In[ ]:


# Save a CSV file, but don't save the index as a column
cs_df.to_csv('ComputerSalesBU.csv', index=False)


# In[ ]:


# You can read data from Excel, but not formulas and macros
pd.read_excel('Financial Sample.xlsx',0)


# In[ ]:


# Write to Excel
cs_df.to_excel('ComputerSales.xlsx')


# In[ ]:


# Check if written
pd.read_excel('ComputerSales.xlsx',0)


# In[ ]:


# Read from MySQL Database
try:
    db_connection = pymysql.connect(db='students', user='studentadmin', passwd='TurtleDove', host='localhost', port=3306)

    stud_df = pd.read_sql('SELECT * FROM students', con=db_connection)
    # print(stud_df)
except Exception as e:
    print("Exception : {}".format(e))
finally:
    db_connection.close()


# In[ ]:


# Write to table 
try:
    db_connection = pymysql.connect(db='students', user='studentadmin', passwd='TurtleDove', host='localhost', port=3306)
    # Used to issue queries
    cursor = db_connection.cursor()
    # Query to enter new student
    insert_stmt = "INSERT INTO students VALUES(NULL, 'Frank', 'Silva', 'fsilva@aol.com', '666 Hell St', 'Yakima', 'WA', 98901, '792-223-8966', '1959-2-22', 'M', NOW(), 3.50)"
    # Execute query
    cursor.execute(insert_stmt)
    # Commit changes to DB
    db_connection.commit()
    stud_df = pd.read_sql('SELECT * FROM students', con=db_connection)
    print(stud_df)
except Exception as e:
    print("Exception : {}".format(e))
finally:
    db_connection.close()


# In[ ]:


# Just get 1 column of data 
cs_df_st = pd.read_csv('ComputerSales.csv', usecols=["State"], squeeze=True)
cs_df_st


# ## Basics Techniques & Maths Operations

# In[ ]:


# Display 1st 5 rows
cs_df.head()


# In[ ]:


# Display last 5 rows
cs_df.tail()


# In[ ]:


# Get 1st 2
cs_df[:2]


# In[ ]:


# Get 1st through 5 with a 2 step
cs_df[:5:2]


# In[ ]:


# Get indexes
cs_df.index.array


# In[ ]:


# Get NumPy array
cs_df.to_numpy()


# In[ ]:


# Get array from series
ser_1.array

dict_3 = {'one': pd.Series([1., 2., 3.], index=['a', 'b', 'c']),
         'two': pd.Series([1., 2., 3., 4.], index=['a', 'b', 'c', 'd'])}
df_2 = pd.DataFrame(dict_3)


# In[ ]:


# You can replace NaN values with 0 or anything else
print(df_2.fillna(0))


# In[ ]:


# Get values in row 2
row = df_2.iloc[1]


# In[ ]:


# Add items in row 2 to all rows including row 2
# You can do the same with sub, mul, and div
df_2.add(row, axis='columns')


# In[ ]:


# Get column 2
col = df_2['two']


# In[ ]:


# Subtract from other columns
df_2.sub(col, axis=0)


# In[ ]:


# Check if empty
df_2.empty


# In[ ]:


# Transform executes a function on a dataframe
df_5 = pd.DataFrame({'A': range(3), 'B': range(1, 4)})
df_5.transform(lambda x: x+1)
df_5.transform(lambda x: x**2)
df_5.transform(lambda x: np.sqrt(x))


# In[ ]:


# You can transform using multiple functions
df_5.transform([lambda x: x**2, lambda x: x**3])


# In[ ]:


# Passing a dictionary allows you to perform different calculations
# on different columns
df_5.transform({'A': lambda x: x**2, 'B': lambda x: x**3})


# In[ ]:


# map performs a function on a series
df_5['A'].map(lambda x: x**2)


# In[ ]:


# applymap does the same on a dataframe
df_5.applymap(lambda x: x**2)


# In[ ]:


# Get unique values in column 2 of DF
df_2['two'].unique()


# In[ ]:


# Get number of uniques
df_2['two'].nunique()


# In[ ]:


# Get the number of times each value showed in column 2
df_2['two'].value_counts()


# In[ ]:


# Get column names
df_2.columns


# In[ ]:


# Get index info
df_2.index


# In[ ]:


# Return a DF that lists null values as True
df_2.isnull()


# ## Group Data

# In[ ]:


# Groupby allows you to group rows based on a columnand perform a function
# that combines those values (Aggregate Function)
dict_5 = {'Store': [1,2,1,2], 'Flavor': ['Choc', 'Van', 'Straw', 'Choc'], 'Sales': [26, 12, 18, 22]}

df_11 = pd.DataFrame(dict_5)


# In[ ]:


# Group data by the store number
by_store = df_11.groupby('Store')


# In[ ]:


# Get mean sales by store
by_store.mean()


# In[ ]:


# Get sales total just for store 1
by_store.sum().loc[1]


# In[ ]:


# You can use multiple functions of get a bunch
by_store.describe()


# ## Concatenate, Merge & Join Data

# In[ ]:


# You can concatenate DFs in the order DFs are provided
df_12 = pd.DataFrame({'A': [1,2,3],'B': [4,5,6]},index=[1,2,3])
df_13 = pd.DataFrame({'A': [7,8,9],'B': [10,11,12]},index=[4,5,6])
pd.concat([df_12, df_13])


# In[ ]:


# Merge 2 DFs using their shared key column
df_12 = pd.DataFrame({'A': [1,2,3],'B': [4,5,6],'key': [1,2,3]})
df_13 = pd.DataFrame({'A': [7,8,9],'B': [10,11,12],'key': [1,2,3]})


# In[ ]:


# inner merges at the intersection of keys
pd.merge(df_12, df_13, how='inner', on='key')


# In[ ]:


# how='left' or 'right' : Use keys from left or right frame
# how='outer' : Use union of keys


# In[ ]:


# You can join DFs with different indexes and instead of using 
# keys use a column
df_12 = pd.DataFrame({'A': [1,2,3],'B': [4,5,6]},index=[1,2,3])
df_13 = pd.DataFrame({'C': [7,8,9],'D': [10,11,12]},index=[1,4,5])
df_12.join(df_13, how='outer')


# ## Statistics

# In[ ]:


# Get ice cream sales data
ics_df = pd.read_csv('icecreamsales.csv')
ics_df


# In[ ]:


# Get total count of both columns
ics_df.count()


# In[ ]:


# skipna skips null / NaN values
ics_df.sum(skipna=True)


# In[ ]:


# Get mean for named column
ics_df["Sales"].mean()
ics_df["Sales"].median()
ics_df["Sales"].mode()
ics_df["Sales"].min()
ics_df["Sales"].max()
ics_df["Sales"].prod() # Product of values
ics_df["Sales"].std() # Standard deviation
ics_df["Sales"].var() # Variance
ics_df["Sales"].sem() # Standard error


# In[ ]:


# Negative : Left long tail, Positive : Right long tail
ics_df["Sales"].skew()


# In[ ]:


# Kurtosis : < 3 less outliers, 3 Normal Distribution, > 3 more outliers
ics_df["Sales"].kurt()
ics_df["Sales"].quantile(.5)
ics_df["Sales"].cumsum()
ics_df["Sales"].cumprod()
ics_df["Sales"].cummax()
ics_df["Sales"].cummin()


# In[ ]:


# Multiple stats at once
ics_df.describe()

ser_dice = pd.Series(data=[2, 3, 3, 4, 4, 4, 5, 5, 5, 5, 6, 6, 
                           6, 6, 6, 7, 7, 7, 7, 7, 7, 8, 8, 8,
                          8, 8, 9, 9, 9, 9, 10, 10, 10, 11, 11, 12])


# In[ ]:


# Count for each value in series
ser_dice.value_counts()


# In[ ]:


# You can perform calculations on multiple columns using aggregate
print(df_2)
df_2.agg(np.mean)


# In[ ]:


# You can do this with multiple functions
df_2.agg(['mean', 'std'])


# ## Iteration

# In[ ]:


# Iterating over series
ser_7 = pd.Series(range(5), index=['a', 'b', 'c', 'd', 'e'])
for col in ser_7:
    print(col)


# In[ ]:


# Iterating over DFs
arr_4 = np.random.randint(10, 50, size=(2, 3))
df_8 = pd.DataFrame(arr_4, ['B', 'C'], ['C', 'D', 'E'])
print(df_8)


# In[ ]:


# items allows you to iterate through key value pairs to make
# calculations 1 column at a time
for label, ser in df_8.items():
    print(label)
    print(ser)


# In[ ]:


# You can also iterate through rows
for index, row in df_8.iterrows():
    print(f"{index}\n{row}")


# In[ ]:


# Get a tuple that contains row data
for row in df_8.itertuples():
    print(row)


# ## Sorting

# In[ ]:


df_8


# In[ ]:


# Sorting by index will return the same results if indexes
# are in order, to reverse indexes mark ascending as False
df_8.sort_index(ascending=False)


# In[ ]:


# Sort by value for column D (Use the same function for series)
df_8.sort_values(by='D')


# ## Passing Data to Functions

# In[ ]:


import sys


# In[ ]:


# You can pass DataFrames and Series into functions
def get_profit_total(df):
    prof_ser = df['Profit']
    print(f"Total Profit : {prof_ser.sum()}")

get_profit_total(cs_df)


# In[ ]:


# Receives a DataFrame, splits the contact into new columns
# being first and last name
def split_name(df):
    def get_names(full_name):
        # Split contact at space
        f_name, l_name = full_name.split()
        # Create a series with first & last names in columns
        # with those labels
        return pd.Series(
        (f_name, l_name),
        index=['First Name', 'Last Name']
        )
    # apply() executes the function on all names in Contact column
    names = df['Contact'].apply(get_names)
    df[names.columns] = names
    return df


# In[ ]:


# Run function and display top 5 results
split_name(cs_df).head()


# In[ ]:


# Will assign people to different age groups based on age
def create_age_groups(df):
    # Must have 1 more bins than labels
    bins = [0, 30, 50, sys.maxsize]
    # Group labels
    labels = ['<30', '30-50', '>50']
    
    # cut puts values into certain groups based on intervals
    # The group assigned to <30 has an age between 0 and 30
    # between 30 & 50 is assigned 30-50 and so on
    age_group = pd.cut(df['Age'], bins=bins, labels=labels)
    # Create new column and return new dataframe info
    df['Age Group'] = age_group
    return df

create_age_groups(cs_df)


# In[ ]:


# You can use a pipe to pass a dataframe to multiple functions
cs_df.pipe(split_name).pipe(create_age_groups).head()


# ## Aligning, Reindexing and Renaming Labels

# In[ ]:


ser_6 = pd.Series(range(5), index=['a', 'b', 'c', 'd', 'e'])
sl_1 = ser_6[:4]
sl_2 = ser_6[1:]
print(sl_1)
print(sl_2)


# In[ ]:


# Align both series by the union of their indexes
sl_1.align(sl_2)


# In[ ]:


# Align by calling series
sl_1.align(sl_2, join='left')


# In[ ]:


# Use passed series indexes
sl_1.align(sl_2, join='right')


# In[ ]:


# Get where indexes intersect
sl_1.align(sl_2, join='inner')


# In[ ]:


# You can use align with DFs as well
arr_3 = np.random.randint(10, 50, size=(2, 3))
df_6 = pd.DataFrame(arr_3, ['A', 'B'], ['C', 'D', 'E'])
arr_3 = np.random.randint(10, 50, size=(2, 3))
df_7 = pd.DataFrame(arr_3, ['B', 'C'], ['C', 'D', 'E'])
df_6
df_6.align(df_7)


# In[ ]:


# reindex allows you to align data by index
ser_6.reindex(['c','b','a'])


# In[ ]:


# Do the same with DFs
df_6.reindex(['B','A'])


# In[ ]:


# Drop is very similar to reindex except it receives labels
# you don't want to include
df_6.drop(['A'], axis=0)
df_6.drop(['D'], axis=1)


# In[ ]:


# You can rename labels
df_6.rename(columns={'C': 'Men', 'D': 'Women', 'E': 'Pets'},
           index={'A': 1, 'B': 2})


# ## MultiIndex

# In[ ]:


# Multi-level indexing allows you to store data on multiple dimensions
days = ['Day 1', 'Day 1', 'Day 1', 'Day 2', 'Day 2', 'Day 2']
meals = [1,2,3,1,2,3]


# In[ ]:


# zip pairs the days and meals arrays 
# Then we create a list of those paired tuples
hier_index = list(zip(days, meals))
print(hier_index)


# In[ ]:


# Converts list of tuples into each row and column
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[ ]:


# Generate random array representing calories eaten per meal
arr_5 = np.random.randint(500, 700, size=(6, 2))
df_9 = pd.DataFrame(arr_5, hier_index, ['M', 'F'])
print(df_9)


# In[ ]:


# Grab the day 1 DF
df_9.loc['Day 1']


# In[ ]:


# Grab 1st row as a series
df_9.loc['Day 1'].loc[1]


# In[ ]:


# Grab calories eaten by the female on day 2 for the 2nd meal
df_9.loc['Day 2'].loc[2]['F']


# In[ ]:


# We can assign names to the Day and Meals Column
df_9.index.names = ['Day', 'Meal']
df_9


# In[ ]:


# Get a cross section
# This gets me the Day 2 DF
df_9.xs('Day 2')


# In[ ]:


# Get calories for the 1st meal for both days by saying what
# meal index you want and the Meal column name
df_9.xs(1, level='Meal')


# In[ ]:


# Create a MultiIndex out of a DF using a pivot table
dict_6 = {'A':['Day 1', 'Day 1', 'Day 1', 'Day 2', 'Day 2', 'Day 2'],
          'B': [1,2,3,1,2,3],
          'C': ['M', 'F', 'M', 'F', 'M', 'F'],
          'D': [1,2,3,4,5,6]}

df_14 = pd.DataFrame(dict_6)


# In[ ]:


# Designate the D column is the data
# Make A & B a multilevel index
# Define column names come from column C
# You will have NaNs where data was missing
df_14.pivot_table(values='D', index=['A','B'], columns=['C'])


# ## Handling Missing Data

# In[ ]:


dict_4 = {'A': [1,2,np.nan], 'B': [4, np.nan, np.nan], 'C': [7.,8.,9.]}
df_10 = pd.DataFrame(dict_4)
print(df_10)


# In[ ]:


# Drop missing data from DF (Drops any row with missing values)
df_10.dropna()


# In[ ]:


# Drop all columns with any missing data
df_10.dropna(axis=1)


# In[ ]:


# Drop row unless it has at least 2 non-NaN values
df_10.dropna(thresh=2)


# In[ ]:


# Fill NaN values with 0
df_10.fillna(value=0.0)


# In[ ]:


# Fill A column with the mean of column
df_10['A'].fillna(value=df_10['A'].mean())


# In[ ]:


# Fill with previous value
df_10.fillna(method='ffill')


# In[ ]:


# Fill with next value (Only works if there is a next value)
df_10.fillna(method='bfill')


# ## Experimenting with Data

# In[ ]:


# Get 1st 5
cs_df.head() 


# In[ ]:


# Get column names
print(cs_df.columns) 


# In[ ]:


# Average profit per item
cs_df['Profit'].mean()


# In[ ]:


# Get the product with the highest profit
cs_df[['Product ID', 'Profit']].max(axis=0).head()


# In[ ]:


# Number of people who purchased from WV
cs_df[cs_df['State']=='WV']['State'].count()


# In[ ]:


# Number of purchases in 2019
len(cs_df[cs_df['Year']==2019].index)


# In[ ]:


# Get number of sales for each product type
cs_df['Product ID'].value_counts()


# In[ ]:


# Get list of customers that bought a specific product
cs_df[cs_df['Product ID']=='M01-F0024']['Contact']


# In[ ]:


# How many made a website purchase for a profit over $200
cs_df[(cs_df['Lead']=='Website') & (cs_df['Profit']>150)]['Lead'].count()


# In[ ]:


# Find out how many product profit amounts include .89 in cents
cs_df['Profit'].apply(lambda cents: str(cents).split('.')[1]=='89').value_counts()


# ## Visualization

# In[ ]:


# Library usef to create advanced static, animated and
# interactive visualizations
import matplotlib.pyplot as plt


# In[ ]:


# Displays matplotlib plots in the Notebook
get_ipython().run_line_magic('matplotlib', 'inline')


# Histograms provide an approximation of the distribution of results. You create them by dividing the range of values into bins or buckets. Then you count how many of the results fall into each bin. Rolls 2 dice 5000 times and charts the frequency and a histogram

# In[ ]:


# Even though the odds increase as you approach 7 and then
# decrease again (1 way to roll a 2 / 6 ways to roll a 7)
# over many rolls they are nearly equal.
df_dice = pd.DataFrame(np.random.randint(1,7,5000),columns = ['Hist'])

df_dice['Odds'] = df_dice['Hist'] + np.random.randint(1,7,5000)


# In[ ]:


# Alpha decreases the opacity in the chart
ax = df_dice.plot.hist(bins=12, alpha=0.5)


# In[ ]:


# Basic plot using 1000 random values that create cumulative sums
# over an increasing date range
ser_5 = pd.Series(np.random.randn(1000), index=pd.date_range('11/15/2017', periods=1000))
ser_5 = ser_5.cumsum()


# In[ ]:


ser_5.plot()


# In[ ]:


# Display 3 random plots
df_15 = pd.DataFrame(np.random.randn(1000, 3),
                    index=pd.date_range('11/15/2017', periods=1000),
                    columns=list('ABC'))
df_15 = df_15.cumsum()


# In[ ]:


df_15.plot()


# In[ ]:


# Make bar chart from 5 random values
# pd.DataFrame(np.random.randn(5)).plot.bar()


# In[ ]:


# Make MultiBar Charts
vals = ['A', 'B', 'C', 'D']
df_15 = pd.DataFrame(np.random.rand(10,4), columns=vals)


# In[ ]:


df_15.plot.bar()


# In[ ]:


# Area plot 
# Define x range and y values
x_rng = range(1,15)
y_vals = [1,5,4,7,6,9,5,7,10,14,10,12,9,8]


# In[ ]:


# Change fill color and opacity
plt.fill_between(x_rng, y_vals, color="skyblue", alpha=0.5)
plt.show()


# In[ ]:


# Area plot with multiple areas
pd.DataFrame(np.random.rand(10,3), columns=['A','B','C']).plot.area()


# In[ ]:


# Create a scatterplot with 100 random values
pd.DataFrame(np.random.rand(100,2), columns=['A','B']).plot.scatter(x='A', y='B')


# In[ ]:


# Multiple column scatter plots
df_15 = pd.DataFrame(np.random.rand(50,4), columns=['A','B','C','D'])


# In[ ]:


ax = df_15.plot.scatter(x='A', y='B', color='DarkBlue', label='Grp 1')

df_15.plot.scatter(x='C', y='D', color='Orange', label='Grp 2', ax=ax)


# In[ ]:


# Pie Charts with 4 random values
pd.Series(np.random.rand(4), index=['a','b','c','d'], name='Pie').plot.pie(figsize=(6,6))

