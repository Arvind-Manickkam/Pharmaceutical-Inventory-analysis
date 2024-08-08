# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 19:24:27 2024

@author: Muthukumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

dir(pd)

# Read data into Python
medical = pd.DataFrame(pd.read_excel(r"C:/Users/Muthukumar/Desktop/Data Set.xlsx"))

medical.info()
medical.describe()
medical.isnull().sum()
medical.nunique()


medical.columns


medical1 = medical[['Material', 'Material Description', 'Plant', 'Storage Location',
       'Movement Type','Posting Date', 'Qtyin_Un_ofEntry','Unit of Entry', 'Movement Type Text',
       'Document Date', 'Qty_in_OPUn', 'Order Price Unit','Order Unit',
       'Qty_in_orderunit', 'Entry Date', 'Amount_in_LC',
       'Purchase Order', 'Movement indicator','Base Unit of Measure', 'Quantity',
       'Material Doc. Year', 'Debit/Credit ind',
        'Trans./Event Type', 'Material Type','Vendor Code']]


medical1
medical1.isnull().sum()
medical1.info()


medical1.mean()


#Missing values

medical1.isnull().sum()
len(medical1)


# To find the Percentage of missing values
medical1.isna().mean().round(4) * 100

#Storage Location         5.12%
#Order Price Unit        59.57%
#Order Unit              59.56%
#Purchase Order          59.42%
#Movement indicator      59.56%

#Checking the values 
medical1['Storage Location'].value_counts()

medical1[medical1['Storage Location'].isna()]

#Filling NAN values

medical1['Storage Location'].fillna('', inplace= True)
medical1.isnull().sum()

############# Data Pre-processing ##############

################ Type casting #################

# Convert categorical variables to appropriate data types
medical1['Movement Type'] = medical1['Movement Type'].astype(str)

medical1.info()


# Exploratory Data Analysis
# Measures of Central Tendency / First moment business decision
medical1.Qtyin_Un_ofEntry.mean() # '.' is used to refer to the variables within object
medical1.Qtyin_Un_ofEntry.median()
medical1.Qtyin_Un_ofEntry.mode()

medical1.Qty_in_OPUn.mean() # '.' is used to refer to the variables within object
medical1.Qty_in_OPUn.median()
medical.Qty_in_OPUn.mode()

medical1.Qty_in_orderunit.mean() # '.' is used to refer to the variables within object
medical1.Qty_in_orderunit.median()
medical1.Qty_in_orderunit.mode()

medical1.Amount_in_LC.mean() # '.' is used to refer to the variables within object
medical1.Amount_in_LC.median()
medical1.Amount_in_LC.mode()

medical1.Quantity.mean() # '.' is used to refer to the variables within object
medical1.Quantity.median()
medical1.Quantity.mode()


medical1.rename(columns={'Posting Date': 'Posting_Date', 'Entry Date': 'Entry_Date', 'Document Date': 'Document_Date'}, inplace=True)
medical1.info()

#Mode for Categorical Columns

medical1.Posting_Date.mode()
medical1.Entry_Date.mode()
medical1.Document_Date.mode()

medical1.Posting_Date.mean()
medical1.Posting_Date.median()

# Measures of Dispersion / Second moment business decision
medical1.Qtyin_Un_ofEntry.var() # variance
medical1.Qtyin_Un_ofEntry.std() # standard deviation
range = max(medical.Qtyin_Un_ofEntry) - min(medical.Qtyin_Un_ofEntry) # range
range

medical1.Qty_in_OPUn.var() # variance
medical1.Qty_in_OPUn.std() # standard deviation
range = max(medical.Qty_in_OPUn) - min(medical.Qty_in_OPUn) # range
range

medical1.Qty_in_orderunit.var() # variance
medical1.Qty_in_orderunit.std() # standard deviation
range = max(medical.Qty_in_orderunit) - min(medical.Qty_in_orderunit) # range
range

medical1.Amount_in_LC.var() # variance
medical1.Amount_in_LC.std() # standard deviation
range = max(medical.Amount_in_LC ) - min(medical.Amount_in_LC ) # range
range

medical1.Quantity.var() # variance
medical1.Quantity.std() # standard deviation
range = max(medical.Quantity) - min(medical.Quantity) # range
range


# Third moment business decision
medical1.Qtyin_Un_ofEntry.skew()
medical1.Qty_in_OPUn.skew()
medical1.Qty_in_orderunit.skew()
medical1.Amount_in_LC.skew()
medical1.Quantity.skew()


# Fourth moment business decision
medical1.Qtyin_Un_ofEntry.kurt()
medical1.Qty_in_OPUn.kurt()
medical1.Qty_in_orderunit.kurt()
medical1.Amount_in_LC.kurt()
medical1.Quantity.kurt()


#Univ/Bivariate/Multivariate Plots

# Data Visualization
import pandas as pd

# Read data into Python
#medical = pd.read_excel(r"C:/Users/Muthukumar/Desktop/Data Set.xlsx")


# Read data into Python
medical1.shape

#Qtyin_Un_ofEntry  Qty_in_OPUn  Qty_in_orderunit  Amount_in_LC  Quantity

# Histogram
plt.hist(medical1.Qtyin_Un_ofEntry) # histogram
plt.hist(medical1.Qtyin_Un_ofEntry, color='red', edgecolor = "black", bins = 5)

plt.hist(medical1.Qty_in_OPUn) # histogram
plt.hist(medical1.Qty_in_OPUn, color='yellow', edgecolor = "black", bins = 5)

plt.hist(medical1.Qty_in_orderunit) # histogram
plt.hist(medical1.Qty_in_orderunit, color='green', edgecolor = "black", bins = 5)

plt.hist(medical1.Amount_in_LC) # histogram
plt.hist(medical1.Amount_in_LC, color='blue', edgecolor = "black", bins = 5)

plt.hist(medical1.Quantity) # histogram
plt.hist(medical1.Quantity, color='orange', edgecolor = "black", bins = 5)

####
help(plt.hist)

#Qtyin_Un_ofEntry  Qty_in_OPUn  Qty_in_orderunit  Amount_in_LC  Quantity
 

# Boxplot
plt.figure()
plt.boxplot(medical1.Qtyin_Un_ofEntry)
plt.boxplot(medical1.Qty_in_OPUn)
plt.boxplot(medical1.Qty_in_orderunit)
plt.boxplot(medical1.Amount_in_LC)
plt.boxplot(medical1.Quantity)


help(plt.boxplot)


# Density Plot
sns.kdeplot(medical1.Qtyin_Un_ofEntry) # Density plot
sns.kdeplot(medical1.Qty_in_OPUn)
sns.kdeplot(medical1.Qty_in_orderunit)
sns.kdeplot(medical1.Amount_in_LC)
sns.kdeplot(medical1.Quantity)

# Descriptive Statistics
# describe function will return descriptive statistics including the 
# central tendency, dispersion and shape of a dataset's distribution.

medical1.describe()


# Bivariate visualization
# Scatter plot
import pandas as pd
import matplotlib.pyplot as plt

#medical = pd.read_excel(r"C:/Users/Muthukumar/Desktop/Mini Project Files/Medical Inventory Optimization Dataset.xlsx")

medical.info()

# Removing Duplicates
medical2 = medical1.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Parameters
medical2 = medical1.drop_duplicates(keep = 'last')

medical2 = medical1.drop_duplicates(keep = False)

medical2

df=medical2


# Countplot to visualize the distribution of Storage Locations within each Plant
plt.figure(figsize=(12, 8))
sns.countplot(data=df, x='Plant', hue='Storage Location')
plt.title('Distribution of Storage Locations within each Plant')
plt.xlabel('Plant')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Storage Location', loc='upper right')
plt.show()

# Crosstab to get the frequency of each Storage Location within each Plant
crosstab = pd.crosstab(df['Plant'], df['Storage Location'])
print("Cross-tabulation of Plant and Storage Location:")
print(crosstab)

# Heatmap to visualize the relationship between Plant and Storage Location
plt.figure(figsize=(12, 8))
sns.heatmap(crosstab, annot=True, cmap='Blues', fmt='d')
plt.title('Relationship between Plant and Storage Location')
plt.xlabel('Storage Location')
plt.ylabel('Plant')
plt.show()

# Scatter plot to visualize the relationship between Amount_in_LC and Sales Value

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Amount_in_LC', y='Sales Value')
plt.title('Relationship between Amount_in_LC and Sales Value')
plt.xlabel('Amount_in_LC')
plt.ylabel('Sales Value')
plt.show()


# Scatter plot to visualize the relationship between Quantity and Qtyin_Un_ofEntry
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Quantity', y='Qtyin_Un_ofEntry')
plt.title('Relationship between Quantity and Qtyin_Un_ofEntry')
plt.xlabel('Quantity')
plt.ylabel('Qtyin_Un_ofEntry')
plt.show()

correlation = df[['Quantity', 'Quantity']].corr()
print("Correlation between Amount_in_LC and Sales Value:")
print(correlation)


# Distribution of Movement Type
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Movement Type')
plt.title('Distribution of Movement Types')
plt.xlabel('Movement Type')
plt.ylabel('Count')
plt.show()

# Distribution of Posting Date
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Posting Date', bins=30)
plt.title('Distribution of Posting Dates')
plt.xlabel('Posting Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Group data by movement type and sum the quantities
grouped_data = df.groupby('Movement Type')['Quantity'].sum().reset_index()

# Plot the bar plot
plt.figure(figsize=(8, 6))
plt.bar(grouped_data['Movement Type'], grouped_data['Quantity'], color='skyblue')
plt.xlabel('Movement Type')
plt.ylabel('Total Quantity')
plt.title('Total Quantity by Movement Type')
plt.show()

#Distribution of Quantity
plt.figure(figsize=(10, 6))
sns.histplot(medical1['Quantity'], bins=20, kde=True)
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')
plt.ylabel('Frequency')
plt.show()

# Visualize relationships
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Posting_Date', y='Amount_in_LC')
plt.title('Amount vs. Posting Date')
plt.xlabel('Posting Date')
plt.ylabel('Amount in Local Currency')
plt.show()


## Correlation Matrix##
# Correlation coefficient

#Ranges from -1 to +1. 
#Rule of thumb says |r| > 0.85 is a strong relation

# Select numerical columns for correlation analysis
numerical_columns = ['Quantity', 'Amount_in_LC','Qtyin_Un_ofEntry','Qty_in_OPUn','Qty_in_orderunit']  # Add more numerical columns as needed

# Subset the data with numerical columns
numerical_data = medical1[numerical_columns]

# Calculate the correlation matrix
correlation_matrix = numerical_data.corr()

print("Correlation Matrix:")
print(correlation_matrix)

#Heatmap

sns.heatmap(correlation_matrix,annot=True)
plt.show()

medical1.head()


### Identify duplicate records in the data ###


# Duplicates in rows
help(medical1.duplicated)

duplicate = medical1.duplicated()  # Returns Boolean Series denoting duplicate rows.
duplicate

sum(duplicate)

# Parameters
duplicate = medical1.duplicated(keep = 'last')
duplicate

duplicate = medical1.duplicated(keep = False)
duplicate


# Removing Duplicates
medical2 = medical1.drop_duplicates() # Returns DataFrame with duplicate rows removed.

# Parameters
medical2 = medical1.drop_duplicates(keep = 'last')

medical2 = medical1.drop_duplicates(keep = False)

medical2

df=medical2

#Mean & Median & Mode after removing duplicates

df.Qtyin_Un_ofEntry.mean() # '.' is used to refer to the variables within object
df.Qtyin_Un_ofEntry.median()
df.Qtyin_Un_ofEntry.mode()

df.Qty_in_OPUn.mean() # '.' is used to refer to the variables within object
df.Qty_in_OPUn.median()
df.Qty_in_OPUn.mode()

df.Qty_in_orderunit.mean() # '.' is used to refer to the variables within object
df.Qty_in_orderunit.median()
df.Qty_in_orderunit.mode()

df.Amount_in_LC.mean() # '.' is used to refer to the variables within object
df.Amount_in_LC.median()
df.Amount_in_LC.mode()

df.Quantity.mean() # '.' is used to refer to the variables within object
df.Quantity.median()
df.Quantity.mode()
