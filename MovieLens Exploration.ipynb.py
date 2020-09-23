#!/usr/bin/env python
# coding: utf-8

# ### `Project - MovieLens Data Analysis`
# 
# The GroupLens Research Project is a research group in the Department of Computer Science and Engineering at the University of Minnesota. The data is widely used for collaborative filtering and other filtering solutions. However, we will be using this data to act as a means to demonstrate our skill in using Python to “play” with data.
# 
# 
# ### `Objective:`
# - To implement the techniques learnt as a part of the course.
# 
# ### `Learning Outcomes:`
# - Exploratory Data Analysis
# 
# - Visualization using Python
# 
# - Pandas – groupby, merging 
# 
# 
# ### `Domain` 
# - Internet and Entertainment
# 
# **Note that the project will need you to apply the concepts of groupby and merging extensively.**

# ### `Datasets Information:`
# 
# 
# *rating.csv:* It contains information on ratings given by the users to a particular movie.
# - user id: id assigned to every user
# - movie id: id assigned to every movie
# - rating: rating given by the user
# - timestamp: Time recorded when the user gave a rating
# 
# 
# 
# *movie.csv:* File contains information related to the movies and their genre.
# - movie id: id assigned to every movie
# - movie title: Title of the movie
# - release date: Date of release of the movie
# - Action: Genre containing binary values (1 - for action 0 - not action)
# - Adventure: Genre containing binary values (1 - for adventure 0 - not adventure)
# - Animation: Genre containing binary values (1 - for animation 0 - not animation)
# - Children’s: Genre containing binary values (1 - for children's 0 - not children's)
# - Comedy: Genre containing binary values (1 - for comedy 0 - not comedy)
# - Crime: Genre containing binary values (1 - for crime 0 - not crime)
# - Documentary: Genre containing binary values (1 - for documentary 0 - not documentary)
# - Drama: Genre containing binary values (1 - for drama 0 - not drama)
# - Fantasy: Genre containing binary values (1 - for fantasy 0 - not fantasy)
# - Film-Noir: Genre containing binary values (1 - for film-noir 0 - not film-noir)
# - Horror: Genre containing binary values (1 - for horror 0 - not horror)
# - Musical: Genre containing binary values (1 - for musical 0 - not musical)
# - Mystery: Genre containing binary values (1 - for mystery 0 - not mystery)
# - Romance: Genre containing binary values (1 - for romance 0 - not romance)
# - Sci-Fi: Genre containing binary values (1 - for sci-fi 0 - not sci-fi)
# - Thriller: Genre containing binary values (1 - for thriller 0 - not thriller)
# - War: Genre containing binary values (1 - for war 0 - not war)
# - Western: Genre containing binary values (1 - for western - not western)
# 
# 
# 
# *user.csv:* It contains information of the users who have rated the movies.
# - user id: id assigned to every user
# - age: Age of the user
# - gender: Gender of the user
# - occupation: Occupation of the user
# - zip code: Zip code of the use
# 
# 
# **`Please provide you insights wherever necessary.`**

# ### 1. Import the necessary packages - 2.5 marks

# In[1093]:


import numpy as np                    # import numpy library for EDA as np
import pandas as pd                   # import Pandas library for EDA as pd
import matplotlib.pyplot as plt       # import matplotlib Library 
import seaborn as sns                 # import Seboarn Library for Data visualisation
import os as os                       # import Operating system Library to run OS commands
os.getcwd()                           # get working directory path to import Datasets

# This command print the graphs in Jypter Notebook without Plt.show() command

get_ipython().run_line_magic('matplotlib', 'inline')


# ### 2. Read the 3 datasets into dataframes - 2.5 marks

# In[1464]:


dfuser = pd.read_csv("C:\\Users\\manogoel\\Downloads\\user.csv")         # import user information Dataset to dataframe 'dfuser'
dfmrating = pd.read_csv("C:\\Users\\manogoel\\Downloads\\Data.csv")      # import Movie Rating Dataset to dataframe 'dfmrating'
dfmlist = pd.read_csv("C:\\Users\\manogoel\\Downloads\\item.csv")        # import Movie List Dataset to dataframe 'dfmlist'


# ### 3. Apply info, shape, describe, and find the number of missing values in the data - 5 marks
#  - Note that you will need to do it for all the three datasets seperately

# # User information Dataset Descriptive Analysis

# In[1465]:


# 3.1.1 User Inforamtion Dataset : Describe and understand Dataset
print(' ## Number of Rows & Colums in User information Dataset : \n')
dfuser.shape


# In[1466]:


## 3.1.2 User Inforamtion Dataset : check the shape of Data i.e datatype for each column, Non-Null Value in each Column
dfuser.info()


# In[1467]:


## 3.1.3 User Inforamtion Dataset :view the Five Point Summary information for each column with nummeric Entries
dfuser.describe().T


# In[1468]:


## 3.1.4 User Inforamtion Dataset : check Non-Null values for each column in Dataset
dfuser.count()


# In[1469]:


## 3.1.5 User Inforamtion Dataset : check if any missing value in Dataset
dfuser.isnull().any()


# # Movie Rating Dataset Descriptive Analysis

# In[1470]:


# 3.2.1 Movie Rating Dataset : Describe and understand Dataset
print(' ## Number of Rows & Colums in Movie Rating Dataset : \n')
dfmrating.shape


# In[1471]:


## 3.2.2 Movie Rating Dataset : check the shape of Data i.e datatype for each column, Non-Null Value in each Column
dfmrating.info()


# In[1472]:


## 3.2.3 Movie Rating Dataset : view the Five Point Summary information for each column with nummeric Entries
dfmrating.describe().T


# In[1473]:


## 3.2.4 Movie Rating Dataset : check Non-Null values for each column in Dataset
dfmrating.count()


# In[1474]:


## 3.2.5 Movie Rating Dataset : check if any missing value in Dataset
dfmrating.isnull().any()


# # Movie List Dataset Descriptive Analysis

# In[1475]:


# 3.3.1 Movie List Dataset : Describe and understand Dataset
print(' ## Number of Rows & Colums in Movie List Dataset : \n')
dfmlist.shape


# In[1476]:


## 3.3.2 Movie List Dataset : check the shape of Data i.e datatype for each column, Non-Null Value in each Column
dfmlist.info()


# In[1477]:


## 3.3.3 Movie List Dataset : view the Five Point Summary information for each column with nummeric Entries
dfmlist.describe().T


# In[1478]:


## 3.3.4 Movie List Dataset : check Non-Null values for each column in Dataset
dfmlist.count()


# In[1479]:


# 3.3.5 Movie List Dataset : check if any missing value in Dataset
dfmlist.isnull().any()


# ### 4. Find the number of movies per genre using the item data - 2.5 marks

# In[1480]:


# check unique Genre List
dfmlist.columns


# In[1481]:


# Movie List Dataset has Genre type attribute in column and Genre field contains Binary Value 
# 1 if movie categorised in that genre, 0 if movie not in that speific genre 
# summing the Genre Value will provide the Movie per Genre Count
# Datset Index 3 to 21 contians Genre Value therefore print only column 3 to 21
# use this Keyword to sort the Movie count by Genre dfmlist.sum()[3:22].sort_values(ascending=True)

# dfmlist.sum()[3:22].sort_values(ascending=True)
# Smart way is create an Array, later import this arany in Pandas Dataframe, Sort by Movie count and Print

arr = {'Movie Count':dfmlist.sum()[3:22]}
df=pd.DataFrame(arr)
df.sort_values('Movie Count',ascending=True)


# In[1482]:


# Get the sum of Genre for all movies using NumPy Library
np.sum(dfmlist.sum()[3:22])


# Observations 
# 1. There are 1681 Movies in List, 
# 2. categorized in 19 Genres, 
# 3. No Duplicate Genre category in the List, 
# 4. Some Movies categorised in more than one Genre
# 5. Total Genre Ratings 2892
# 6. 1 Movie Genre is Unknown

# ### 5. Drop the movie where the genre is unknown - 2.5 marks

# In[1483]:


# To Drop the Movies with Unknown Genre, sort the MovieList Dataframe in Descending by Genre Unknown
# Import this in new Dataframe 'dfmlist1'
# check the count of Movies with unknown Genre
dfmlist[dfmlist.unknown ==1].sum()


# In[1484]:


# Validate only one moview with Unknown Genre by Sorting the Movie List
# Print the Head to see entries, Match the Index ID and other details from Previous command output

dfmlist1=dfmlist.sort_values([('unknown')], ascending=False)

# alternatively you can use this command to print all movies with unknow genre by using head() 
# dfmlist[dfmlist.unknown ==1].head() 

dfmlist1.head()


# Observation : Only one movie in the List with Unknwon Genre, listed on index No 1371, Movie ID 1373

# In[1485]:


# Drop the unknow genre entries from Dataframe
# Drop Sepcific Entry by using below command, advisable to drop using condition, 
# Store all value in Datafarme 1 where Unknow Genre is Not Equal to 1
# dfmlist1 = dfmlist1.drop([1371])

dfmlist1 = dfmlist[dfmlist.unknown != 1]


# In[1486]:


# Sort the New Dataframe by Unknown Genre and print Head to see if any unknown genre Exist. 
dfmlist1.sort_values([('unknown')], ascending=False).head()


# In[1487]:


# Validate the Size of new Dataframe,should have 1 entry less from initial dataframe
dfmlist1.shape


# Observation : Unknown Genre Movie dropped from the Data frame 
# only one entry in the dataset with unknown Genre found, New Dataset size is 1680

# ### 6. Find the movies that have more than one genre - 5 marks
# 
# hint: use sum on the axis = 1
# 
# Display movie name, number of genres for the movie in dataframe
# 
# and also print(total number of movies which have more than one genres)

# In[1488]:


dfmlist2=dfmlist1.copy()    # Import the dataframe in new Dataframe "dfmlist2"

# Add new colum "Genre Count" in new DataFrame, 
# Fill the Value in new Datafarem by summing the columns 3 to 21 

dfmlist2['Genre Count']=dfmlist1.iloc[:,3:22].sum(axis=1)
dfmlist2.head().T      #check new colum added and providing the Genre count for each movie


# In[1489]:


# Display movie name, number of genres for the movie in dataframe, I undestood we need original Moview List Datafrae for this Excercise

dfmlist3=dfmlist2[['movie title','Genre Count']]   #import the Movie Title & Genre count in new Data Frame
dfmlist3.set_index('movie title',inplace=True)     # Set Movie title as row Index Values
dfmlist3                                           # Display Dataframe Movie Titel & Genre Count


# In[1490]:


# print(total number of movies which have more than one genres)
df4=dfmlist2[dfmlist2['Genre Count'].values>1].count()
print("More than one Genre ",df4[['movie title']])


# Observation : 849 Movies in the Movie list has more than one Genre

# ### 7. Univariate plots of columns: 'rating', 'Age', 'release year', 'Gender' and 'Occupation' - 10 marks
# 
# *HINT: Use distplot for age. Use lineplot or countplot for release year.*
# 
# *HINT: Plot percentages in y-axis and categories in x-axis for ratings, gender and occupation*
# 
# *HINT: Please refer to the below snippet to understand how to get to release year from release date. You can use str.split() as depicted below or you could convert it to pandas datetime format and extract year (.dt.year)*

# In[1491]:


a = 'My*cat*is*brown'
print(a.split('*')[3])

#similarly, the release year needs to be taken out from release date

#also you can simply slice existing string to get the desired data, if we want to take out the colour of the cat

print(a[10:])
print(a[-5:])


# Movie Ratings Analysis

# In[1492]:


# seaborn library is imported in start
# Univariate Plot for Rating
sns.countplot(data=dfmrating, x='rating')    #Use simple Countplot for count of ratings


# User Age Analysis

# In[1493]:


# Univariate Plot for Age, USer given movie rating belong to which agen group
x=dfuser['age']         # import the age colum to varibale x, dfuser is user dataframe
sns.distplot(x)         # You can also print same graph using sns.distplot(dfuser['age'])


# Release Year Analysis

# In[1494]:


dfmlist4=dfmlist2.copy()                                                                  # import movie list dataset to dfmy
dfmlist4['release date']=pd.to_datetime(dfmlist4['release date'],format='%d-%b-%Y') # Covert the release date Object type to date Data type

# dfmlist4.info()     Check the Relase Date Data type


# In[1495]:


# Extract Movie Year from Release Data and add to new Column Year in Dataframe               
dfmlist4['R Year']=pd.DatetimeIndex(dfmlist4['release date']).year
dfmlist4.head()       # Verify new Colum Year is updated 


# In[1496]:


ax=sns.countplot(data=dfmlist4, x='R Year')                               #Use simple Countplot for Yearly movie Count
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")   # Format the x Axis Label rotate 90 Degree
plt.show()


# In[1497]:


# Yearly Movie Release not readable in simple count plot, print distplot to see normalize movie relase in 10 Years slot

x=dfmlist4['R Year']         # import the age colum to valibale x, dfuser is user dataframe
sns.distplot(x)         # You can also print same graph using sns.distplot(dfuser['age'])


# Moview Reviewer's Gender Analysis

# In[1498]:


sns.countplot(data=dfuser, x='gender')    #Use simple Countplot for count of Gender


# Movie Reviewer's Occupation Analysis

# In[1499]:


ax=sns.countplot(data=dfuser, x='occupation')                       #Use simple Countplot for count of Occupation
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")   # Format the x Axis Label rotate 90 Degree
plt.show()


# ### 8. Visualize how popularity of genres has changed over the years - 10 marks
# 
# Note that you need to use the **percent of number of releases in a year** as a parameter of popularity of a genre
# 
# Hint 1: You need to reach to a data frame where the release year is the index and the genre is the column names (one cell shows the number of release in a year in one genre) or vice versa. (Drop unnecessary column if there are any)
# 
# Hint 2: Find the total number of movies release in a year(use `sum(axis=1)` store that value in a new column as 'total'). Now divide the value of each genre in that year by total to get percentage number of release in a particular year.
# `(df.div(df['total'], axis= 0) * 100)`
# 
# Once that is achieved, you can either use univariate plots or can use the heatmap to visualise all the changes over the years 
# in one go. 
# 
# Hint 3: Use groupby on the relevant column and use sum() on the same to find out the number of releases in a year/genre.  

# # My Solution steps 
# 1. Copy the Dataframe to newdataframe
# 2. drop the unwanted column, Keep only Genre and movie Release Year
# 3. Add New column and fill the values in colum as 1, basically this is movie release count
# 4. group by dataframe on Movie Release Year and get the sum of all columns
# 5. Now the new Data farme has unique Year, total movie release in that year by Genre

# In[1500]:


dfmlist5=dfmlist4.copy()
dfmlist5=dfmlist5.drop(['movie id','movie title','release date','unknown','Genre Count'],axis=1)
dfmlist5['Yearly Movie Release']=1
dfmlist5=dfmlist5.groupby(['R Year']).sum()
dfmlist5


# 6. Get the % release in each genre category by diving total movie release in the year

# In[1501]:


dfmlist6=dfmlist5.div(dfmlist5['Yearly Movie Release'], axis= 0) * 100
dfmlist6=dfmlist6.drop(['Yearly Movie Release'],axis=1)
dfmlist6.style.set_precision(0)
dfmlist6


# In[1502]:


sns.heatmap(dfmlist6, annot=False)


# In[1503]:


cname = dfmlist6.columns

for n, cname in enumerate(dfmlist6.columns):
    ax=sns.distplot(dfmlist6[cname])
    #ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="right")
    plt.show()


# Result : 
# 1. Dramam & Comedy Movies are always one of the Poular category
# 2. Animation, Western not a poluplar Genre

# ### 9. Find the top 25 movies according to average ratings such that each movie has number of ratings more than 100 - 10 marks
# 
# Hints : 
# 
# 1. Find the count of ratings and average ratings for every movie.
# 2. Slice the movies which have ratings more than 100.
# 3. Sort values according to average rating such that movie which highest rating is on top.
# 4. Select top 25 movies.
# 5. You will have to use the .merge() function to get the movie titles.
# 
# Note: This question will need you to research about groupby and apply your findings. You can find more on groupby on https://realpython.com/pandas-groupby/.

# # Steps i follow
# 1. copy the Movie rating dataset to new Dataframe
# 2. Add another colum to get Movie Average Rating
# 3. Use Groupby function on movie id to get the rating sum and feedback count
# 4. Make average Rating by sum of Rating / Rating count
# 5. Drop Unwanted columns and copy to new dataframe

# In[1504]:


dfmrating1=dfmrating.copy()

dfmrating1['M Rating Count']=1
dfmrating2=dfmrating1.groupby(['movie id']).sum()
dfmrating2['Avg Rating'] = dfmrating2['rating'].div(dfmrating2['M Rating Count'], axis= 0)
dfmrating10=dfmrating2.copy()                                  # For final Merger of the Datbase
dfmrating2=dfmrating2.drop(['user id','timestamp'],axis=1)
dfmrating2.head()


# # Steps i Follow
# 6. slice the data by etracting moving rating more than 100

# In[1505]:


dfmrating3 = dfmrating2.copy()
dfmrating3=dfmrating3[dfmrating3.rating >100]
dfmrating3.head()


# # Steps i Follow
# 
# 7. Extract Movie ID & Movie Title from Movile list Dataframe to new Datfarm 10
# 8. Set the moveie id as Index (not mandatory)
# 9. Merge the two dataframe to get Movie title updated in extracted Movie Rating List
# 10. Sort the movies in Descending Averge Rating Order
# 11. Extract Top 25 Movies from to new Dataframe
# 12. check the top 25 moview by issues dftop25.shape command (should include only 25 record)

# In[1506]:


dfmlist10=dfmlist[['movie id','movie title']]
dfmlist10.set_index('movie id')
dfmrating4 = pd.merge(dfmlist10,dfmrating3, how='right', on='movie id')
dfmrating4 = dfmrating4.sort_values([('Avg Rating')], ascending=False)
dftop25=dfmrating4.head(25)
dftop25.head(25)


# # Result : 
# 1. dftop25 dataframe include top 25 movies meeting criteria
# 2. Rating for each movie is more than 100
# 3. dataframe only include top 25 Movies by Avereg Rating 

# ### 10. See gender distribution across different genres check for the validity of the below statements - 10 marks
# 
# * Men watch more drama than women
# * Women watch more Sci-Fi than men
# * Men watch more Romance than women
# 
# **compare the percentages**

# 1. Merge all the datasets
# 
# 2. There is no need to conduct statistical tests around this. Just **compare the percentages** and comment on the validity of the above statements.
# 
# 3. you might want ot use the .sum(), .div() function here.
# 
# 4. Use number of ratings to validate the numbers. For example, if out of 4000 ratings received by women, 3000 are for drama, we will assume that 75% of the women watch drama.

# In[1535]:


#dfmrating2.head()

df_usr_rat = pd.merge(dfuser,dfmrating1, how='right', on='user id')
df_usr_rat.head()
df_ml=pd.merge(df_usr_rat,dfmlist4,how='right',on='movie id')
df_ml.head()


# In[1552]:


df_ml2=df_ml.drop(['user id','age','occupation','zip code','movie id','timestamp','movie title','release date','Genre Count','R Year'],axis=1)
df_ml2.head()


# In[1553]:


df_ml3=df_ml2.copy()
df_ml3=df_ml2.groupby(['gender']).sum()
df_ml3.head()


# In[ ]:


df_ml4=df_ml3.copy()
df_ml4=df_ml3.div(df_ml3['M Rating Count'], axis= 0) * 100
df_ml4.head().T


# In[ ]:


sns.pairplot(df_ml4)
plt.show()


# Observation : All below Observations are Right
# 1. Men watch more drama than women
# 2. Women watch more Sci-Fi than men
# 3. Men watch more Romance than women
