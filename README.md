# Table of Contents
1. [Introduction](#introduction)
2. [Data Description](#data-description)
3. [Key Challenges](#key-challenges)
4. 
5. 
6. 
7. 
8. 
9. [Limitations](#limitations)
10. 
11. 
12. 
13. 
14. [Project Structure](#capstone-project-structure)

## Introduction  
In 2006, Netflix released over 100 million ratings from over 400,000 users on over 17,000 films. Whoever improved the RMSE of Netflix’s “Cinematch” algorithm by 10% within 5 years received $1,000,000. Here, we revisit this contest by using a subset of this dataset, with a different test set than the original contest.

With the rise of social media, recommendation algorithms are more important than ever. Even children and non-technical people throw around the term "The Algorithm," showing its prescence in the current zeitgeist. Despite this challenge being annouced in 2006, I chose this project as I believed the skills I would learn from it would be especially relevant in the current landscape.

The goal of this project is to improve this RMSE by as much as possible using an ensemble of an SVD, K nearest neighbors, and baseline algorithm. This project was assigned during finals season to simulate working conditions; to see what I can produce when multiple projects are demanding my attention. 

Though sources online differ, the RMSE (root mean square error) of the base "Cinematch" algorithm falls somewhere in the range of 0.9474 to 0.9525. With my limited resources, both time and compute, surpassing this baseline I would consider a success (spoiler - RMSE achieved: 0.93409). 

## Data Description  
I was given two files: 'data.txt' and 'movieTitles.csv'. The file 'data.txt' is a plaintext file containing movie ids and rating information (the user id, a rating 1-5, and date rated for each review). It is over 27 million lines long, containing ratings on 5k movies from ~400k users, and the data is rather sparse as most users haven't rated most movies. Due to the size and formatting of the data, I created a special function in 'src/parsing.py' called "load_ratings()" that reads the file line by line and converts the results into a pandas dataframe. I then use another function to save the dataframe as a parquet in 'data/processed/ratings.parquet'. I did this to improve efficiency when repeatedly loading the data and to reduce the amount of storage the data took up.

The file 'movieTitles.csv' contains a table of each movie id, its release year, its corresponding movie title. It is a csv file that is 5000 lines long. Because of the commas in the movie titles, I wrote a function that manually parses each column and returns the data as a dataframe. 

I then created a train-test split by using *all* ratings for a given movie in the training set, except for *one* randomly picked rating. That one rating (per movie) constitutes the test set. This was done with the "train_test" function in 'src/preprocessing.py'.

## Key Challenges  
### Sparsity  
Since most users haven't rated most movies, the data is very sparse, i.e. many user-movie combinations do not exist in the data. With sparse data, certain models aren't as well suited and overfitting becomes even more of a concern. My approach to this problem involved
- Using aggregation for the baseline models which reduces variance via pooling
- Residual modeling, in the user/movie/time bias model, capturing systematic over/underestimation trends common with sparse data
- SVD, designed to take large, sparse matrices and decompose them into smaller, denser matrices capturing important latent features
### Cold-Start  
A problem with recommendation systems is the cold-start problem -- what do you do when there is little to no information on a user/movie? I handled this in my code by
- Using the global mean for movies with no reviews (if they exist)
- Using 0 user bias for new users
- Using the mean movie rating for pairs of new users and new movies
### Bias/Variance
To balance bias and variance in my data, I used
- Regularization terms that shrink the impact of movies with few ratings
- Filtering out users with few (<20) ratings for time bias calcualations

## Limitations
The data used in this project is fairly limited. Ideally, there would be information on genre, language, and other metadata that could be incorporated to make better predictions. Additionally, I was rather limited by computing power with my models taking a total of 1hr 20min to train. This made it not feasible to implement cross validation, though if I could've, I would've.

## Project Structure
### Project contents
* README.md
* pyproject.toml
  * Setup for this project
  * Run [pip install -e .] to install packages
* .vscode/
  * settings.json
    * Settings for VSCode
* .gitignore
  * Ignore data and configurations
* data/
  * raw/
    * data.txt
      * Contains movie rating information
    * movieTitles.csv
      * Contains movie titles and IDs
  * processed/
    * ratings.parquet
      * A parquet containing the parsed data.txt file
* notebooks/
  * exploration.ipynb
    * Used for exploring data before implementing
* src/
  * \_\_init__.py
    * Treat this directory as a package
  * config.py
    * N-number and paths to data
  * parsing.py
    * Getting the data into the dataframes and parquet
  * preprocessing.py
    * Getting the data into a more workable state
  * baseline.py
    * Train baseline models
  * svd.py
    * Train svd model
  * knn.py
    * Train K nearest neighbors model
  * evaluation.py
    * Evaluate the models
* img/
  * Images
* models/
  * svd_model.pkl
  * knn_model.pkl
    * Files containing the trained models (since they take a long time to train)
* predictions/
  * bias_model_preds.parquet
  * global_mean_preds.parquet
  * movie_mean_preds.parquet
  * knn_preds.parquet
  * svd_preds.parquet
  * time_bias_model_preds.parquet
    * Contains the predictions made by each model
* main.py
  * Main python file