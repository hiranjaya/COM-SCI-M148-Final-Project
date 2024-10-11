import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from data_cleaning.py import clean(df) 

df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv")
revised_df = clean(df)

# Look at linear correlations between features
def plot_corr_matrix(df):
  corr_matrix = df.corr(method='pearson')
  plt.figure(figsize=(10 , 10))
  sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
  plt.show()

# Plot some scatterplots of features with higher correlation
def plot_scatter(df)
  sample = df.sample(n=1000, random_state=42)

  fig,ax = plt.subplots(2,2,figsize=(10,10))

  ax[0,0].scatter(sample['energy'], sample['acousticness'])
  ax[0,0].set_xlabel('Energy')
  ax[0,0].set_ylabel('Acousticness')

  ax[0,1].scatter(sample['loudness'], sample['acousticness'])
  ax[0,1].set_xlabel('Loudness')
  ax[0,1].set_ylabel('Acousticness')

  ax[1,0].scatter(sample['loudness'], sample['instrumentalness'])
  ax[1,0].set_xlabel('Loudness')
  ax[1,0].set_ylabel('Instrumentalness')

  ax[1,1].scatter(sample['valence'], sample['danceability'])
  ax[1,1].set_xlabel('valence')
  ax[1,1].set_ylabel('danceability')
  plt.show()

# Plot a boxplot to look at popularity by the key of the song
def plot_box_plot(df)
  df.boxplot(column='popularity', by='key', grid=False)
  plt.xlabel('Key')
  plt.ylabel('Popularity')

plot_corr_matrix(revised_df)
plot_scatter(df)
plot_box_plot(df)
