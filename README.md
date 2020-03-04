# K-NN Movie Recommender

The aim of this project is to recommend movies to a user by using a user-based collaborativefiltering approach. For this project a data set of 100,000 movie ratings provided by [MovieLens](https://grouplens.org/datasets/movielens/100k/), consisting of 943 users and 1682 movies was used. This data setwas used to find users with similar movie rating in order to recommend film for the user to watch.


## K Nearest Neighbours
In order to recommend film to a user k-nearest neighbours (k-NN) was used. K-NN is a supervised machine learning algorithm used for classification and regression, however it is mostly used for classification problems.  The code to achieve this was written in Python using a combination of Pandas and Sklearn. At first the data is loaded from the files provided by the MovieLens data set into to Pandas dataFrames to make the data simpler to work with. 

## Method
Once the rating data is loaded into the a dataFrame it is then passed into a pivot table with the index, columns and values set to user id, film id and rating respectively. If the user hasn't provided a rating for a given film id the rating is set to zero. The pivot table is then converted into a sparse matrix using the scipy python module. The sparse matrix of all the data can then be passed into k-NN using cosine similarity distance metric and brute-force algorithm. In cosine similarity each user is represented as vector. Cosine similarity subtracts 1 from the cosine of the angle between the two vectors in order to calculate the distance between two users. The distance of between the two vectors can then be used to determine how similar the users are to each other. The closer the distance is to 0 the higher the similarity is between the users. Cosine similarity is quick to compute for sparse matrix as only none-zero elements are considered. Brute-force was the chosen algorithm as it is the simplest to implement but for large data sets in can be computationally expensive. Brute-force is adequate for this application as this data set is relatively small. 


After finding the most similar users to the initial user, the films that both users have watched are compared.  If the neighbouring user has watched a movie that the initial user has not watched, that film is added to an un-watched movie list. This process is then completed for all the nearest neighbours. Only films that similar users have rated 4 or 5 are considered, as films with less than a rating of 4 are assumed to be not of interest to the user.  Once the un-watched movies list has been completed all movies that only occur once are removed, leaving only films that have been suggested by more than one neighbour. This method was chosen as if more than one nearest neighbour suggests the same film, the certainty that the film is a good recommendation for the initial user is higher. Next a rating that the initial user may give for each of the films is predicted. The score is predicted by taking a weighted mean of the neighbours' rating whom have rated the film. The weighting is based on the distance to the initial user. The closer the distance the higher the weighting the user's rating has. 

  
The next stage was to calculate the optimum k value for the data set. To calculate the performance of each k value the root mean square error (RMSE) and mean absolute error (MAE) was calculated. To calculate the RSME and MAE the data set was split into a train and test set with an 80\%/20\% split respectively. As each k value was ran, the predicted score for the recommend films was compared against the actual score given by the user in the test data set. A range of k values between 2 to 100 was tried to find the optimum k value, Figure 1. From the results it can be seen that in the k value range from 15 to 18 the RSME and MAE reach a global minima. Upon closer inspection of the results it was found that a k value of 17 is the optimum value for this data set. If the data set was to change a new k values must be assessed. 

![Figure 1: Comparison of MAE/RAE for different k values](https://i.imgur.com/c8CXxVM.jpg | width=500)
**Figure 1: Comparison of MAE/RAE for different k values**

## Testing
To prove that the user-based movie recommendation system provides an actual recommendation, the performance of the system must be evaluated using the RSME and MAE values. Two simulation methods were performed; random and worst-case. To keep the testing consistent between the different methods the k values was always set to 17. For the random simulation, each predicted rating was randomly generated from a uniform distribution. For the worst-case simulation, the system was ran to maximise the RSME and MAE value to simulate a worst-case scenario. This was achieved by setting the predicted rating to furthest possible value from the test data in order to maximise the difference between the predicted rating and the test rating. It was found that the RSME and MAE of the developed system are lower than the valves from the random and worst-case simulation, Figure 2. The developed system has a 43.5\% improvement over random in RSME value and 45.6\% in MAE. This shows that the system is providing actual movie recommendations as it is better than a random guess. 

![Figure 2: Comparison of MAE/RAE for rating prediction methods - k=17](https://i.imgur.com/Mv1yahA.jpg | width=500)
**Figure 2: Comparison of MAE/RAE for rating prediction methods - k=17**

## Improving Accuracy Using User Details
The next stage was to assess the impact that of adding users' details to improve the accuracy of the recommendations. Provided in the data set is the age, gender, occupation and zip-code of each user. Both gender and occupation data are provided in character form and must be converted into integers in order to pass it through the k-NN algorithm. Gender was converted to 1 for M and 1000 for F, to create a large difference between the genders to reduce the likely hood of opposite genders being considered similar. Occupation data was converted into integer data by indexing each occupation with its alphabetical position in the occupation list.  
  
With the data converted a series of tested was run to find the combination of user details that provide the best performance. Each combination was tried with a k value of 17, and it's RSME and MAE value was calculated, Figure 3. From the tests it was found the combining gender and occupation with the rating data provided the best results. A 3.9\% increase in RSME accuracy and 4.3\% increase in MAE. This change is very small, and it can be assumed that the rating alone provides sufficient recommendations.

![Figure 3: Comparison of user details on MAE/RAE - k=17](https://i.imgur.com/XMqKsjF.jpg | width=500)
`<img src="https://i.imgur.com/XMqKsjF.jpg" width="500">`
**Figure 3: Comparison of user details on MAE/RAE - k=17**


