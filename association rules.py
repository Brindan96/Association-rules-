# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 14:39:33 2020

@author: Hp
"""

# Association Rules 

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
book=pd.read_csv("F:\\Data Science\\Assignemnts\\Brindan\\association rules\\book.csv")
sb.countplot(book.ChildBks)

# apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules

# min support =0.1
books = apriori(book,min_support=0.1, max_len=3,use_colnames = True)
books.shape # 39  

# min support =0.01
books = apriori(book,min_support=0.01, max_len=3,use_colnames = True)
books.shape # 208  

# min support =0.005
books = apriori(book,min_support=0.005, max_len=3,use_colnames = True)
books.shape # 224  

# Most Frequent item sets based on support 
books.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = books.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),books.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(books, metric="lift", min_threshold=1)
rules.shape

rules.sort_values('lift',ascending = False,inplace=True)

def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)







# groceries

with open("F:\\Data Science\\Assignemnts\\Brindan\\association rules\\groceries.csv","r") as f:
    groceries = f.read()
    
groceries1 = []

# splitting the data into separate transactions using separator as "\n"
groceries1 = groceries.split("\n")

groceries_list = []
for i in groceries1:
    groceries_list.append(i.split(","))
    
    
all_groceries_list = []

#for i in groceries_list:
#    all_groceries_list = all_groceries_list+i
    
    
all_groceries_list = [i for item in groceries_list for i in item]
from collections import Counter

item_frequencies = Counter(all_groceries_list)
# after sorting

item_frequencies = sorted(item_frequencies.items(),key = lambda x:x[1])


# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 

import matplotlib.pyplot as plt

plt.bar(height = frequencies[:11],x = list(range(0,11)),color='rgbkymc');plt.xticks(list(range(0,11),),items[:11]);plt.xlabel("items")
plt.ylabel("Count")


# Creating Data Frame for the transactions data 
import pandas as pd

groceries_series  = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835,:] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep='*').str.get_dummies(sep='*')

frequent_itemsets = apriori(X,min_support=0.005, max_len=3,use_colnames = True)
frequent_itemsets.shape


# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = frequent_itemsets.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),frequent_itemsets.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.shape


rules.head(20)
rules.sort_values('lift',ascending = False,inplace=True)

 
# To eliminate Redudancy in Rules 
def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)







# My movies

movies=pd.read_csv("F:\\Data Science\\Assignemnts\\Brindan\\association rules\\My_movies.csv")    
movies.isna().sum()
movies.columns
movies.drop(["V1","V2","V3","V4","V5"], inplace=True,axis = 1)


# apriori algorithm
from mlxtend.frequent_patterns import apriori,association_rules

# min support =0.1
movies1 = apriori(movies,min_support=0.1, max_len=3,use_colnames = True)
movies1.shape # 46 

# min support =0.01
movies2 = apriori(movies,min_support=0.2, max_len=3,use_colnames = True)
movies2.shape # 13 rows 

# min support =0.005
movies3 = apriori(movies,min_support=0.3, max_len=3,use_colnames = True)
movies3.shape # 7 rows 

# Most Frequent item sets based on support 
books.sort_values('support',ascending = False,inplace=True)
plt.bar(x = list(range(1,11)),height = movies1.support[1:11],color='rgmyk');plt.xticks(list(range(1,11)),movies1.itemsets[1:11])
plt.xlabel('item-sets');plt.ylabel('support')

rules = association_rules(movies1, metric="lift", min_threshold=1)
rules.shape

rules.sort_values('lift',ascending = False,inplace=True)

def to_list(i):
    return (sorted(list(i)))


ma_X = rules.antecedents.apply(to_list)+rules.consequents.apply(to_list)


ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]
index_rules = []
for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))


# getting rules without any redudancy 
rules_no_redudancy  = rules.iloc[index_rules,:]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift',ascending=False).head(10)
