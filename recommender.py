## Find nearest neighbours to a user and return recommended films based on
## what the nearest neighbours enjoyed. Predict a rating that the user would
## give for the recommended films by calculating the weighted mean of the
## ratings given by the neighbours; weighting is the distance from the user.

import collections
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Load relevant files into dataFrame
film = pd.read_csv('ml-100k/u.item', sep='|', encoding='ISO-8859-1')
film.columns = ['filmId', 'title', 'release date', 'video release date',
              'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
              'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
              'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
               'Sci-Fi', 'Thriller', 'War', 'Western']
columns = ['release date', 'video release date',
          'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
          'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
           'Sci-Fi', 'Thriller', 'War', 'Western']
film = film.drop(columns, axis=1)

rating = pd.read_csv('ml-100k/u1.base', sep='\t', encoding='ISO-8859-1')
rating.columns = ['userId', 'filmId', 'rating', 'timestamp']
rating = rating.drop(['timestamp'], axis=1)

# Created a nested dictionary of each user with the filmIds and the ratings
# they provided
watched = collections.defaultdict(dict)
for i in rating.values.tolist():
    watched[i[0]][i[1]] = i[2]

# Create a pivot table with index as userId, columns as filmId, values as rating
rating_pivot = rating.pivot(index='userId', columns = 'filmId',\
                    values='rating').fillna(0)
# Convert the pivot table into a sparse matrix
rating_matrix = csr_matrix(rating_pivot.values)

# Initialise k nearest neighbours
knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
knn.fit(rating_matrix)

k = 17

while True:
    # Get user input for the user id
    user = int(input('User id:'))
    user_index = user - 1

    # Find nearest neighbours
    distances, indices = knn.kneighbors(rating_pivot.iloc[user_index, :]\
                        .values.reshape(1, -1), n_neighbors = k)

    # Films the user has watched
    user_watched = set(watched[rating_pivot.index[user_index]])

    neighbours_watched = {}

    # Print neighbours and their distance from the user
    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Closest users to user {}:\n'.format(rating_pivot.index[user_index]))

        else:
            print('{0}: {1} - distance: {2}'.format(i, rating_pivot.index[indices.flatten()[i]], distances.flatten()[i]))

        neighbours_watched[rating_pivot.index[indices.flatten()[i]]] = watched[rating_pivot.index[indices.flatten()[i]]].copy()

        # Save information in order to calculate predicted rating
        for key, v in neighbours_watched[rating_pivot.index[indices.flatten()[i]]].items():
            neighbours_watched[rating_pivot.index[indices.flatten()[i]]][key] = [1 - distances.flatten()[i], v]
    print('----\n')

    unwatched_films = []
    for u in neighbours_watched:
        a = neighbours_watched[u].keys() - user_watched.intersection(neighbours_watched[u].keys())
        for f in a:
            unwatched_films.append(f)

    # Find unwatched films that are common among neighbours
    common_unwatched = [item for item, count in collections.Counter(unwatched_films).items() if count > 1]

    # Predict rating the user would give for the unwatched films
    common_unwatched_rating = []
    for f in common_unwatched:
        m = []
        w = []

        for u in neighbours_watched:
            if neighbours_watched[u].get(f) is not None:
                m.append(neighbours_watched[u].get(f)[0]*neighbours_watched[u].get(f)[1])
                w.append(neighbours_watched[u].get(f)[0])

        common_unwatched_rating.append([np.sum(m)/np.sum(w), f])
    common_unwatched_rating = sorted(common_unwatched_rating, reverse=True)

    print('10 best recommendations based on what similar users liked:\n')
    for f in common_unwatched_rating[:10]:
        print('{0} - {1} - {2:.2f}'.format(f[1], film.loc[film['filmId'] == f[1]]['title'].values[0], f[0]))
    print('-----\n')
