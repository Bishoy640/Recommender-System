from builtins import sorted

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn

###################################################################
# RECCOMMENDER SYSTEM


df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])
movies = pd.read_csv('Movie_Id_Titles')

df = pd.merge(df,movies, on='item_id')
data = df.copy(deep=True)
df.drop('timestamp', axis=1, inplace=True)
df.drop('item_id', axis=1, inplace=True)

ratings = df.groupby('title')['rating'].mean()
count = df.groupby('title')['rating'].count().sort_values(ascending=False)

full_data = pd.merge_ordered(ratings,count, on='title').sort_values('rating_y', ascending=False).reset_index()
full_data.drop('index', inplace=True, axis=1)
full_data.rename(columns={'rating_x': 'rating', 'rating_y': 'num_of_rating'}, inplace=True)

# Visualize the relation between rating and num_of_rating column (OPTIONAL)
sns.jointplot(data=full_data, x='rating', y='num_of_rating')
plt.show()
moviemat = pd.pivot_table(data=data, columns='title', index='user_id', values='rating')
def getCorrelated(title):
    user_ratings = moviemat[title]
    similar_movie = moviemat.corrwith(user_ratings)
    similar_movie = pd.DataFrame(similar_movie, columns=['correlation'])
    similar_movie.dropna(inplace=True)
    similar_movie = similar_movie.join(count, on='title')
    similar_movie.rename(columns={'rating': 'num_of_ratings'}, inplace=True)
    #Filter where num rating > 100
    similar_movie = similar_movie[similar_movie['num_of_ratings'] > 100]
    return similar_movie.sort_values(ascending=False, by='correlation')

print(getCorrelated('Star Wars (1977)'))







