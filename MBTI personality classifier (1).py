
# coding: utf-8

# # MBTI Personality Classifier
# 
# This programme will classify people into mbti personality types based on their past 50 posts on social media using the basic naivebayesclassifier

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
import string
from nltk.classify import NaiveBayesClassifier


# ### Importing the dataset

# In[2]:


data_set = pd.read_csv("mbti_1.csv")
data_set.tail()


# ### Checking the dataset for missing values

# In[3]:


data_set.isnull().any()


# ## Exploring the dataset

# The size of the dataset

# In[4]:


data_set.shape


# Explroing the posts in posts field

# In[5]:


data_set.iloc[0,1].split('|||')


# Finding the number of posts

# In[6]:


len(data_set.iloc[1,1].split('|||'))


# Finding the unique vales from type of personality column

# In[7]:


types = np.unique(np.array(data_set['type']))
types


# The total number of posts for each type

# In[8]:


total = data_set.groupby(['type']).count()*50
total


# Graphing it for better visualization

# In[9]:


plt.figure(figsize = (12,6))

plt.bar(np.array(total.index), height = total['posts'],)
plt.xlabel('Personality types', size = 14)
plt.ylabel('Number of posts available', size = 14)
plt.title('Total posts for each personality type')


# ## Organising the data to create a bag words model

# Segrating all the posts by their personality types and ***creating a new dataframe to store all this in***

# In[10]:


all_posts= pd.DataFrame()
for j in types:
    temp1 = data_set[data_set['type']==j]['posts']
    temp2 = []
    for i in temp1:
        temp2+=i.split('|||')
    temp3 = pd.Series(temp2)
    all_posts[j] = temp3


# In[11]:


all_posts.tail()


# ### Creating a function to tokenize the words

# In[12]:


useless_words = nltk.corpus.stopwords.words("english") + list(string.punctuation)
def build_bag_of_words_features_filtered(words):
    words = nltk.word_tokenize(words)
    return {
        word:1 for word in words \
        if not word in useless_words}


# A random check of the function

# In[13]:


build_bag_of_words_features_filtered(all_posts['INTJ'].iloc[1])


# ## Creating an array of features

# In[14]:


features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    features += [[(build_bag_of_words_features_filtered(i), j)     for i in temp1]]


# Because each number of personality types have different number of posts they must be splitted accordingle. Taking 80% for training and 20% for testing

# In[15]:


split=[]
for i in range(16):
    split += [len(features[i]) * 0.8]
split = np.array(split,dtype = int)


# In[16]:


split


# Data for training

# In[17]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[18]:


sentiment_classifier = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[19]:


nltk.classify.util.accuracy(sentiment_classifier, train)*100


# Creating the test data

# In[20]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[21]:


nltk.classify.util.accuracy(sentiment_classifier, test)*100


# # The model performs at efficieny of only 10% which is pretty bad.
# 
# ## Hence, instead of selecting all 16 types of personalitys as a unique feature I explored the dataset further and decided to simplify it.
# 
# The Myers Briggs Type Indicator (or MBTI for short) is a personality type system that divides everyone into 16 distinct personality types across 4 axis:
# 
# - Introversion (I) – Extroversion (E)
# - Intuition (N) – Sensing (S)
# - Thinking (T) – Feeling (F)
# - Judging (J) – Perceiving (P)
# <br><br>
# We will use this and create 4 classifyers to classify the person 

# ## Creating a classifyer for Introversion (I) and Extroversion (E)
# 
# **Note:** The details for the steps over here are same as the ones while creating the model above, hence I will only explain the changes

# In[22]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('I' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'introvert')         for i in temp1]]
    if('E' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'extrovert')         for i in temp1]]


# Data for training

# In[23]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[24]:


IntroExtro = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[25]:


nltk.classify.util.accuracy(IntroExtro, train)*100


# Creating the test data

# In[26]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[27]:


nltk.classify.util.accuracy(IntroExtro, test)*100


# Seeing that this model has good somewhat good results, I shall repeat the same with the rest of the traits

# ## Creating a classifyer for Intuition (N) and Sensing (S)

# In[28]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('N' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Intuition')         for i in temp1]]
    if('E' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Sensing')         for i in temp1]]


# Data for training

# In[29]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[30]:


IntuitionSensing = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[31]:


nltk.classify.util.accuracy(IntuitionSensing, train)*100


# Creating the test data

# In[32]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[33]:


nltk.classify.util.accuracy(IntuitionSensing, test)*100


# ## Creating a classifyer for Thinking (T) and Feeling (F)

# In[34]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('T' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Thinking')         for i in temp1]]
    if('F' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Feeling')         for i in temp1]]


# Data for training

# In[35]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[36]:


ThinkingFeeling = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[37]:


nltk.classify.util.accuracy(ThinkingFeeling, train)*100


# Creating the test data

# In[38]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[39]:


nltk.classify.util.accuracy(ThinkingFeeling, test)*100


# ## Creating a classifyer for Judging (J) and Percieving (P)

# In[40]:


# Features for the bag of words model
features=[]
for j in types:
    temp1 = all_posts[j]
    temp1 = temp1.dropna() #not all the personality types have same number of files
    if('J' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Judging')         for i in temp1]]
    if('P' in j):
        features += [[(build_bag_of_words_features_filtered(i), 'Percieving')         for i in temp1]]


# Data for training

# In[41]:


train=[]
for i in range(16):
    train += features[i][:split[i]] 


# Training the model

# In[42]:


JudgingPercieiving = NaiveBayesClassifier.train(train)


# Testing the model on the dataset it was trained for accuracy

# In[43]:


nltk.classify.util.accuracy(JudgingPercieiving, train)*100


# Creating the test data

# In[44]:


test=[]
for i in range(16):
    test += features[i][split[i]:]


# Testing the model on the test dataset which it has never seen before

# In[45]:


nltk.classify.util.accuracy(JudgingPercieiving, test)*100


# # Summarizing the results of the models
# ***
# 

# In[46]:


temp = {'train' : [81.12443979837917,70.14524215640667,80.03456948570128,79.79341109742592], 'test' : [58.20469312585358,54.46262259027357,59.41315234035509,54.40549600629061]}
results = pd.DataFrame.from_dict(temp, orient='index', columns=['Introvert - Extrovert', 'Intuition - Sensing', 'Thinking - Feeling', 'Judging - Percieiving'])
results


# Plotting the results for better appeal

# In[47]:


plt.figure(figsize = (12,6))

plt.bar(np.array(results.columns), height = results.loc['train'],)
plt.xlabel('Personality types', size = 14)
plt.ylabel('Number of posts available', size = 14)
plt.title('Total posts for each personality type')


# In[48]:


labels = np.array(results.columns)

training = results.loc['train']
ind = np.arange(4)
width = 0.4
fig = plt.figure()
ax = fig.add_subplot(111)
rects1 = ax.bar(ind, training, width, color='royalblue')

testing = results.loc['test']
rects2 = ax.bar(ind+width, testing, width, color='seagreen')

fig.set_size_inches(12, 6)
fig.savefig('Results.png', dpi=200)

ax.set_xlabel('Model Classifying Trait', size = 18)
ax.set_ylabel('Accuracy Percent (%)', size = 18)
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(labels)
ax.legend((rects1[0], rects2[0]), ('Tested on a known dataframe', 'Tested on an unknown dataframe'))
plt.show()


# # Testing the models to predict my trait my feeding few of my quora writings
# 
# link to my quora answers feed: https://www.quora.com/profile/Divya-Bramhecha

# Defining a functions that inputs the writings, tokenizes them and then predicts the output based on our earlier classifiers

# In[192]:


def MBTI(input):
    tokenize = build_bag_of_words_features_filtered(input)
    ie = IntroExtro.classify(tokenize)
    Is = IntuitionSensing.classify(tokenize)
    tf = ThinkingFeeling.classify(tokenize)
    jp = JudgingPercieiving.classify(tokenize)
    
    mbt = ''
    
    if(ie == 'introvert'):
        mbt+='I'
    if(ie == 'extrovert'):
        mbt+='E'
    if(Is == 'Intuition'):
        mbt+='N'
    if(Is == 'Sensing'):
        mbt+='S'
    if(tf == 'Thinking'):
        mbt+='T'
    if(tf == 'Feeling'):
        mbt+='F'
    if(jp == 'Judging'):
        mbt+='J'
    if(jp == 'Percieving'):
        mbt+='P'
    return(mbt)
    


# ### Building another functions that takes all of my posts as input and outputs the graph showing percentage of each trait seen in each posts and sums up displaying your personality as the graph title
# 
# **Note:** The input should be an array of your posts

# In[243]:


def tellmemyMBTI(input, name, traasits=[]):
    a = []
    trait1 = pd.DataFrame([0,0,0,0],['I','N','T','J'],['count'])
    trait2 = pd.DataFrame([0,0,0,0],['E','S','F','P'],['count'])
    for i in input:
        a += [MBTI(i)]
    for i in a:
        for j in ['I','N','T','J']:
            if(j in i):
                trait1.loc[j]+=1                
        for j in ['E','S','F','P']:
            if(j in i):
                trait2.loc[j]+=1 
    trait1 = trait1.T
    trait1 = trait1*100/len(input)
    trait2 = trait2.T
    trait2 = trait2*100/len(input)
    
    
    #Finding the personality
    YourTrait = ''
    for i,j in zip(trait1,trait2):
        temp = max(trait1[i][0],trait2[j][0])
        if(trait1[i][0]==temp):
            YourTrait += i  
        if(trait2[j][0]==temp):
            YourTrait += j
    traasits +=[YourTrait] 
    
    #Plotting
    
    labels = np.array(results.columns)

    intj = trait1.loc['count']
    ind = np.arange(4)
    width = 0.4
    fig = plt.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, intj, width, color='royalblue')

    esfp = trait2.loc['count']
    rects2 = ax.bar(ind+width, esfp, width, color='seagreen')

    fig.set_size_inches(10, 7)
    
    

    ax.set_xlabel('Finding the MBTI Trait', size = 18)
    ax.set_ylabel('Trait Percent (%)', size = 18)
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(0,105, step= 10))
    ax.set_title('Your Personality is '+YourTrait,size = 20)
    plt.grid(True)
    
    
    fig.savefig(name+'.png', dpi=200)
    
    plt.show()
    return(traasits)
        


# # Importing my quora answers from a text file
# 
# I copied all my answer from the link i provided before (i broke down the paragraphs as separte posts)

# In[244]:


My_writings = open("Myquora.txt")
my_writing = My_writings.readlines()
#my_writing


# In[245]:


my_posts = my_writing[0].split('|||')
len(my_posts)
#my_posts


# # Using the classifier to predict my personality type

# In[246]:


trait=tellmemyMBTI(my_posts, 'Divy')


# # Concluding note
# 
# My profile according to https://www.16personalities.com/ is INTJ.
# 
# I am pretty happy that using such a basic model it was pretty close to my real profile, only 1 different. And even that difference was very close, between 10% inaccuary which pretty good.
# 
# Although, I am not sure how the classifier will perform on all test cases in general. Specially, the data for some profiles was very less.

# # Sanaya profile

# In[247]:


My_writings = open("Sanayapoem.txt")
my_writing = My_writings.readlines()
#my_writing


# In[248]:


my_posts = my_writing[0].split('|||')
len(my_posts)
#my_posts


# In[249]:


trait = tellmemyMBTI(my_posts,'sanaya')


# # Valentin Pyataev

# In[250]:


My_writings = open("Valentin pyatev.txt")
my_writing = My_writings.readlines()
#my_writing


# In[251]:


my_posts = my_writing[0].split('|||')
len(my_posts)
#my_posts


# In[252]:


trait=tellmemyMBTI(my_posts,'Valentin')


# # MIT gurukul people

# In[253]:


My_writings = open("All texts.txt")
my_writing = My_writings.readlines()
a =[''];
for i in my_writing:
    a[0]=a[0]+i
len(a)


# In[254]:


my_posts = a[0].split('&&&')
len(my_posts)
#my_posts


# Posts for each person

# In[255]:


alls = [None]*len(my_posts)
for i in range(len(my_posts)):
    alls[i] = my_posts[i].split('|||') 


# Email ID connection

# In[256]:


Names = open("Names.txt")
names = Names.readlines()
#names


# In[257]:


for i in range(len(names)):
    names[i] = names[i].replace('@gmail.com\n','')
    print(names[i])
names[len(names)-1]=names[len(names)-1].replace('@gmail.com','')


# In[258]:


for i in range(len(alls)):
    trait=tellmemyMBTI(alls[i],names[i])


# In[260]:


trait

