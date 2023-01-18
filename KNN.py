import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# pd.set_option('display.float_format', lambda x: '%.3f' % x)

# uitlezen van CSV bestanden
rating_df = pd.read_csv('ratings.csv', usecols=['userId', 'cardId', 'leeftijd', 'rating'],
                        dtype={'userId': 'Int64', 'cardId': 'Int64', 'leeftijd': 'Int64', 'rating': 'int32'}, sep=';')
card_df = pd.read_csv('cards.csv', usecols=['cardId', 'title', 'trefwoorden'],
                      dtype={'cardId': 'Int64', 'title': 'str', 'trefwoorden': 'str'}, sep=';')

# Samenvoegen van de CSV bestanden tot één tabel
df = pd.merge(rating_df, card_df, on='cardId')
pd.options.display.width = 0

combine_card_rating = df.dropna(axis=0, subset=['title'])
card_ratingCount = (combine_card_rating.
groupby(by=['title'])['rating'].
count().
reset_index().
rename(columns={'rating': 'totalRatingCount'})
[['title', 'totalRatingCount']]
)

rating_with_totalRatingCount = combine_card_rating.merge(card_ratingCount, left_on='title', right_on='title',
                                                         how='left')

# zetten van de parameters
query_index = 14
popularity_threshold = 60

rating_popular_card = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

pd.set_option('display.min_rows', 40)

card_features_df = rating_popular_card.pivot_table(index='title', columns='userId', values='rating').fillna(0)

card_features_df_matrix = csr_matrix(card_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(card_features_df_matrix)

distances, indices = model_knn.kneighbors(card_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=5)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for {0}:\n'.format(card_features_df.index[query_index]))

    else:
        print('{0}: {1}, with distance of {2}'.format(i, card_features_df.index[indices.flatten()[i]],
                                                      distances.flatten()[i]))
