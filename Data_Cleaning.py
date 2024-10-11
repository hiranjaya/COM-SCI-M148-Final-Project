import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

print(df.columns)
print("*************************************")
print(df.head(10))
print("*************************************")
print(df.tail(10))
print("*************************************")
print(df.describe())
print("*************************************")
print(df.info())
print(df.shape)

for col in df.columns:
    print(col, ": ", df[col].nunique())

# # unique track_ids should be equal to the number of tracks, but it isn't
# Seems like some track_ids show up multiple times with different track_genres (possibly some of the other features are different as well, but that hasn't been confirmed yet)
# Seems like 1 track_genre disappears when the duplicates of the track_ids are removed

# Drop all duplicates (need to remove the first column because these are just indices)
revised_df = df.drop(columns='Unnamed: 0').drop_duplicates(subset=['track_id'])

# Double check that everything lines up
for col in revised_df.columns:
    print(col, ": ", revised_df[col].nunique(), "; Type: ", revised_df[col].dtypes)
print(revised_df.shape)

revised_df.drop(columns=['track_id', 'artists', 'album_name', 'track_name'], inplace=True)
columns = revised_df.columns
le = LabelEncoder()
revised_df['track_genre'] = le.fit_transform(revised_df['track_genre']) # Numerically encode 'track_genre'

scaler = StandardScaler()
revised_df = scaler.fit_transform(revised_df) # Standardize the scale of all features
revised_df = pd.DataFrame(revised_df, columns=columns)
