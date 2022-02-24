# Recommender System Project

Project for the [Recommender System Competition](https://www.kaggle.com/c/recommender-system-2021-challenge-polimi/discussion) at Politecnico di Milano.  


## **Goal**

The application domain is TV programs recommendation. The datasets we provide contains both interactions between users and TV shows, as well as features related to the shows. The main goal of the competition is to discover which items (TV shows) a user will interact with.
Each TV show (for instance, "The Big Bang Theory") can be composed by several episodes (for instance, episode 5, season 3). The goal of the recommender system is not recommend a specific episode, but to recommend the TV show.

## **Description**

The datasets includes around 6.2M interactions, 13k users, 18k items (TV shows) and four feature categories: 8 genres, 213 channels, 113 subgenres and 358k events (episode ids).
The training-test split is done via random holdout, 85% training, 15% test.
The goal is to recommend a list of 10 potentially relevant items for each user. MAP@10 is used for evaluation. You can use any kind of recommender algorithm you wish e.g., collaborative-filtering, content-based, hybrid, etc. written in Python.

---

## Credits

The project was developed using the implementations of the course framework. <br>
The framework source code has been imported in this repository and available in extended form [here](https://github.com/MaurizioFD/RecSys_Course_AT_PoliMi).

## Author

- Luca Minotti ([@lucagrammer](https://github.com/lucagrammer))

---

#### Leaderboard at 9 december 2021
<img src="./leaderboard 9dic.jpg">

#### Final score: 3.3/5

