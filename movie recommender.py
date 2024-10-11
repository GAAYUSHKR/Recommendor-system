
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel,cosine_similarity
from nltk.stem.snowball import SnowballStemmer


# In[64]:


df=pd.read_csv("file:///C:/Users/Avinash/Downloads/the-movies-dataset/movies_metadata.csv")


# In[65]:


print(df.isnull().sum())
print(df.dtypes)
print(df.shape)
df.head(10)


# In[66]:


def count(x):
    y=x['vote_average']
    y1=x['id']
    words=[]
    for i in range(len(x['vote_average'])):
        if y[i]>=6:
            words.append(y1[i])
        i=i+1
    return words


# In[67]:


z=count(df)


# In[48]:


df.head(10)


# In[68]:


df['genres'] = df['genres'].fillna('[]').apply(literal_eval).apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[69]:


df.head()


# In[352]:


vote_count=df[df['vote_count'].notnull()]['vote_count'].astype('int')
vote_average=df[df['vote_average'].notnull()]['vote_average'].astype('int')
c=vote_average.mean()
c


# In[71]:


df['year']=pd.to_datetime(df['release_date'],errors='coerce').apply(lambda x:str(x).split('-')[0] if x!=np.nan else np.nan)


# In[72]:


m=vote_count.quantile(0.95)
m


# In[73]:


qualified=df[(df['vote_count']>=m) & (df['vote_count'].notnull()) & (df['vote_average'].notnull())][['year','genres','popularity','vote_count','vote_average','title']]


# In[74]:


qualified['vote_count']=qualified['vote_count'].astype('int')
qualified['vote_average']=qualified['vote_average'].astype('int')


# In[75]:


s = df.apply(lambda x: pd.Series(x['genres']),axis=1).stack().reset_index(level=1, drop=True)


# In[76]:


s.name='genres'
df1=df.drop('genres',axis=1).join(s)


# In[77]:


df1


# In[78]:


def recomend(genres,percentile=0.95):
    df2=df1[df1['genres']==genres]
    vote_count=df2[df2['vote_count'].notnull()]['vote_count'].astype('int')
    vote_average=df2[df2['vote_average'].notnull()]['vote_average'].astype('int')
    c=vote_average.mean()
    m=vote_count.quantile(percentile)
    
    qualified=df2[(df2['vote_count']>=m) & (df2['vote_count'].notnull()) & (df2['vote_average'].notnull())][['title','year','popularity','vote_count','vote_average','genres']]
    qualified['vote_count']=qualified['vote_count'].astype('int')
    qualified['vote_average']=qualified['vote_average'].astype('int')
    
    qualified['wr']=qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average'])+(m/(m+x['vote_count'])*c),axis=1)
    #ualified['wr'] = qualified.apply(lambda x: (x['vote_count']/(x['vote_count']+m) * x['vote_average']) + (m/(m+x['vote_count']) * C), axis=1)
    qualified=qualified.sort_values('wr',ascending=False).head(250)
    
    return(qualified)


# In[79]:


recomend('Romance').head(15)


# In[80]:


cr=pd.read_csv("file:///C:/Users/Avinash/Downloads/the-movies-dataset/credits.csv")
ky=pd.read_csv("file:///C:/Users/Avinash/Downloads/the-movies-dataset/keywords.csv")


# In[21]:


ln=pd.read_csv("file:///C:/Users/Avinash/Downloads/the-movies-dataset/links.csv")
ra=pd.read_csv("file:///C:/Users/Avinash/Downloads/the-movies-dataset/ratings.csv")


# In[81]:


cr['id']=cr['id'].astype('int')
ky['id']=ky['id'].astype('int')


# In[82]:


df = df.drop([19730, 29503, 35587])


# In[83]:


df['id']=df['id'].astype('int')


# In[84]:


df=df.merge(cr,on='id')
df=df.merge(ky,on='id')


# In[85]:


df.shape


# In[87]:


df = df[df['id'].isin(z)]


# In[88]:


df.shape


# In[33]:


#df['description']=df['overview']+df['tagline']
#df['description']=df['description'].fillna('')


# In[35]:


#tf=TfidfVectorizer(analyzer='word',ngram_range=(1,2),stop_words='english',min_df=0)


# In[36]:


#tfidf_matrix=tf.fit_transform(df['description'])


# In[37]:


#tfidf_matrix.shape


# In[59]:


#cosine=linear_kernel(tfidf_matrix,tfidf_matrix)


# In[89]:



df['cast']=df['cast'].apply(literal_eval)
df['crew']=df['crew'].apply(literal_eval)
df['keywords']=df['keywords'].apply(literal_eval)


# In[90]:


def get_director(x):
    for i in x:
        if i['job']=='Director':
            return i['name']
        else:
            return np.nan


# In[92]:


df['director']=df['crew'].apply(get_director)


# In[93]:


df['cast'] = df['cast'].fillna('[]').apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[94]:


df['cast']=df['cast'].apply(lambda x: x[:3] if len(x)>=3 else x)


# In[95]:


df['keywords'] = df['keywords'].fillna('[]').apply(lambda x: [i['name'] for i in x] if isinstance(x, list) else [])


# In[96]:


df['cast']=df['cast'].apply(lambda x:[str.lower(i.replace(" ","")) for i in x])


# In[105]:


df.head()


# In[104]:


df['director'] = df['director'].astype('str').apply(lambda x: str.lower(x.replace(" ", "")))


# In[107]:


s = df.apply(lambda x: pd.Series(x['keywords']),axis=1).stack().reset_index(level=1, drop=True)
s.name = 'keyword'


# In[108]:


s=s.value_counts()


# In[109]:


s[:5]


# In[117]:


j=s[s>10]


# In[119]:


stemmer=SnowballStemmer('english')


# In[120]:


def filter_keywords(x):
    word=[]
    for i in x:
        if i in j:
            word.append(i)
    return word


# In[122]:


df


# In[123]:


df['keywords']=df['keywords'].apply(filter_keywords)


# In[125]:


df['keywords']=df['keywords'].apply(lambda x:[stemmer.stem(i) for i in x])


# In[126]:


df['keywords'] = df['keywords'].apply(lambda x: [str.lower(i.replace(" ", "")) for i in x])


# In[127]:


df


# In[128]:


df['keywords']=df['keywords'].astype('str')
df['cast']=df['cast'].astype('str')
df['genres']=df['genres'].astype('str')
df['director']=df['director'].astype('str')


# In[129]:


df['soup']=df['keywords']+df['cast']+df['director']+df['genres']


# In[131]:


count = CountVectorizer(analyzer='word',ngram_range=(1, 2),min_df=0, stop_words='english')
count_matrix = count.fit_transform(df['soup'])


# In[132]:


count_matrix.shape


# In[133]:


sim=cosine_similarity(count_matrix,count_matrix)


# In[176]:


sim.shape


# In[250]:


df.columns


# In[287]:


df = df.reset_index()
titles = df['title'].apply(lambda x:str.lower(x))
indices = pd.Series(df.index, index=df['title'].apply(lambda x:str.lower(x)))


# In[200]:


titles[6690]


# In[337]:


def get_recommendations(title):
    idx = indices[title]
    sim_scores = list(enumerate(sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:31]
    movie_indices = [i[0] for i in sim_scores]
    movies1 = df.iloc[movie_indices][['title', 'genres', 'cast', 'year', 'director','popularity']]
    return movies1.head(20)


# In[349]:


#indices['the infinity']


# In[351]:


get_recommendations('finding nemo')

