# Perceptron implementation for UFC fight results prediction
<p align="center">
  <img src="/Images/thetrilogy.jpg" />
</p>

# Overview

This project will implement the training and testing of machine learning algorithms for UFC fight results prediction. The hypothesis is that the model can reach at least 80% of accuracy in it’s predictions. When the training and testing phase are  complete the model will be tested live in UFC 246 Poirier vs Mcgregor on July 10th 2021 9:00 PM CDT 


# Goals

The main goal of this project is to prove the hypothesis that it is possible to reach an accuracy of 80% in UFC fight results prediction with a group of selected features described in the ‘project’ section.


## Steps to reach the goal



1. Extract, Transform and Load a dataset for the model.
2. Develop the model
3. Obtain results and compare with our hypothesis.


# Theoretical framework

The UFC (Ultimate Fighter Championship) is the most famous league of MMA (Mixed Martial Arts) in the world. Based in Las Vegas, Nevada, the institution looks for the best MMA fighters in the world. The league is divided by two main groups: Male and female. And in each, the following weight divisions:



* Men
    * Flyweight
    * Bantamweight
    * Featherweight
    * Lightweight
    * Welterweight
    * Middleweight
    * Light heavyweight
    * Heavyweight
* Women
    * Strawweight
    * Flyweight
    * Bantamweight
    * Featherweight

Every fighter in the UFC aspires to win and retain a division belt and proclaim to be the champion of the division. 



## The rules

Each combat is 1 vs 1 of the same category, in an octagonal cage, within a number of 5 minute rounds. The number of rounds depends on the bout: 3 rounds for regular bouts, 5 rounds for championship bouts. 


### Ways of winning

There are 6 different ways of winning each combat.



* Knock out
* Technical Knockout
* Submission
* Decision Victory
* No contest
* Forfeit



## MMA

MMA is considered a highly graphic fighting sport, not recommended for the faint of heart. MMA includes several martial arts, so a single fighter should know how to strike and submit their opponent. It’s origins come from vale tudo. Some of the most common  fighting styles available in the MMA are:



* Boxing
* Judo
* Kick boxing / Muai Tai
* Wrlesting
* Brazilian Jiu Jitsu
* Karate
* Tae Kwon Do

MMA bouts are usually considered for the betting industry as part of the sports betting category. This has brought attention to data scientist of the world to make prediction models for the win options and possibilities.


## 


## Terminology

Category/Division: The weight category a certain fighter is from.

Corner: The place of the octagon where each fighter starts the fight. The octagon has a blue corner and red corner. These are also the places where the fighter’s staffs are placed.

Dataset: Group of data organized and separated by features (columns) and rows (samples)

Decision victory: A way of winning by the decision of the judges if the match comes to an end without another way of winning. 

Dif (As used in dataframe column titles): Difference between blue fighter minus red fighter.

Forfeit: Forfeit is perhaps the rarest path to victory for a fighter. It occurs when a fighter chooses to no longer continue in the scheduled bout that is underway.

Knockout: When a fighter makes another fighter lose consciousness for a period of time

MMA: Mixed Martial Arts.

Normalization: The action to normalize a group of data for its manipulation.

Artificial Neural Network (ANN):, usually simply called neural network (NN), is a computing system inspired by the [biological neural networks](https://en.wikipedia.org/wiki/Biological_neural_network) that constitute animal [brains](https://en.wikipedia.org/wiki/Brain).

No contest: A type of match result. It comes into play when a fighter refuses to follow the unified rules.

Tapping: When a fighter gives up they tap their opponent or, in some situations, the floor. This normally occurs when the fighter is being submitted.

Technical Knockout (TKO): When the referee considers a fighter unable to continue the fight due to strikes received.

Streak: A consecutive series of events. eg: Lose streak, win streak.

Strike: A directed physical attack with a part of the human body

Submission:  A [combat sports](https://en.wikipedia.org/wiki/Combat_sports) term for yielding to the opponent, and hence resulting in an immediate defeat.

UFC: Ultimate Fighter Championship.


# 


# Project


## The dataset

The dataset was found in kaggle as the name of “Ultimate UFC dataset”. It includes 137 columns and the information of 4,771 official UFC fights. It’s size is 2 megabytes and It consists of 4 files:



* most-recent-event.csv: **The information of the fights in the most recent event.
* task-dummy.csv:** The dataset documentation doesn’t specify why is this csv file for. It contains the name of the fighters for the fights in the next event and their probabilities.
* ufc-master.csv:**The master dataset with the data of 4,771 fights. 
* Upcoming-event.csv:** The data of the fights for the upcoming event.

The csv to be used for the training and testing process will be ‘ufc-master.csv’ because here is where the history of fights are. The ‘upcoming-event.csv’ dataset will be used to test the model on-field. 

This is the completeness of the dataset. Each black section is a value. Each white section is a the absence of value.


![heat_map](/Images/heat.png)


The dataset may be found in the following link:

[https://www.kaggle.com/mdabbert/ultimate-ufc-dataset](https://www.kaggle.com/mdabbert/ultimate-ufc-dataset)


## Hypothesis

Feature Engineering


### 1- Feature selection

The following 15 featuress as inputs for the mode were chosen:



* lose_streak_dif: (Blue lose streak) - (Red lose streak)
* winstreakdif: (Blue win streak) - (Red win streak)
* longest_win_streak_dif: (Blue longest win streak) - (Red longest win streak)
* win_dif: (Blue wins) - (Red wins)
* loss_dif: (Blue losses) - (Red losses)
* total_round_dif: (Blue total rounds fought) - (Red total rounds fought)
* total_title_bout_dif: (Blue number of title fights) - (Red number of title fights)
* ko_dif: (Blue wins by KO/TKO) - (Red wins by KO/TKO)
* sub_dif: (Blue wins by submission) - (Red wins by submission)
* height_dif: (Blue height) - (Red height) in cms
* reach_dif: (Blue reach) - (Red reach) in cms
* age_dif: (Blue age) - (Red age)
* sig_str_dif: (Blue sig strikes per minute) - (Red sig strikes per minute)
* avg_sub_att_dif: (Blue submission attempts) - (Red submission attempts)
* avg_td_dif: (Blue TD attempts) - (Red TD attempts)

The main characteristic they have in common is they describe the difference between the red fighter and the blue fighter in several aspects like career records and physical stats.

The following group of charts represents the distribution of each feature. 


![Feature_distribution](/Images/distribution.png)


_‘Height_dif’_ and _‘reach_dif’ _have a negative skew because of four outliers. These are maybe typos since it would require a 6 meter tall human being to accept a height difference of more than 400 cms. Further in this document the reader will realize these features didn’t have any impact on the model whatsoever.

The following image describes the completeness of the selected features. They are 100% complete.



![filtered_heat_map](/Images/heat2.png)



### 2- Winner feature value replacement

The_ ‘Winner’_ column describes who won the fight (Red/Blue. Since the perceptron neural network produces a boolean output, this column had it’s values replaced; all_ ‘Red’ _values to 1 and all _‘Blue’ _values to 0.

3- Normalization

Due to the nature of the implemented machine learning tool for this project, a normalization process was required. The normalization was made under the l2 norm. The data’s distribution remains intact and now all values are placed between -1 and 1.

4- Training and testing samples

In order for the model to be trained a sample was needed. The dataset was separated randomly in two different bunches:



1. Training sample (4000 rows)
2. Testing sample (771 remaining rows)

These samples were generated uniformly at random without replacement.



Model building

The machine learning technique for the model training and testing is  the multi-layer perceptron artificial neural network. As the present document described, there are 15 inputs which are the 15 selected features from the master dataset. There will be only one output, a boolean number which describes the predicted winner (1 for red fighter, 0 for blue fighter).

The library used in this project was entirely created by the author, the only mainstream machine learning method that was used in this project is sklearn.preprocessing.normalize. It was used in the events described in the 3rd section of the feature engineering section of this document.



![neural_network](/Images/NN.png)



Model development

For the model to be trained it is required to specify the following parameters:



* X: The dataset of our inputs (Learning patterns)
* Wh: Initial hidden weights. An array of numerical values (L X input quantity)
* Wo: Initial output weights. An array of numerical values (output quantity X L)
* L: Hidden neurons in the artificial neural network.
* E: Error index. The gap of error the model is going to reach in order to stop.
* A: a index
* Alpha: alpha index.

Once the model was ready to be trained, several combinations of parameters were required. Unfortunately, after about one hundred different combinations the model developed a prediction between 51% and 58%. In some cases the prediction was more than 80%, but in those cases the predictions stayed in 0.5 instead of 1 and 0. We can interpret this as the model wasn’t trained in these specific cases.

In most of these cases a gap between 5 and 10 hidden neurons were needed for the model to perform effectively. More than 10 neurons tended to overfit. Less than 5 neurons tended to overfit also.

The image below shows the pie charts of the right and wrong predictions in both: testing and training. Since the results for the testing sample were not meeting the project goal, the training sample was also tested to see if it has the same behaviour. It was.

  

<p align="center">
  <img src="/Images/pie1.png" />
</p>




In an effort to meet a better prediction the author decided to select subgroups of the input feature group. These subgroups can be generalized in 3 main categories:



1. Career record
    1. lose_streak_dif
    2. winstreakdif
    3. Longest_win_streak_dif
    4. Win_dif
    5. Loss_dif
    6. Total_round_dif
    7. Total_title_bout_dif
    8. Ko_dif
    9. Sub_dif
2. Physical stats
    10. Height_dif
    11. Reach_dif
    12. Age_dif
    13. Sig_str_dif
    14. Avg_sub_att_dif
    15. avg_td_dif
3. Features with the most spreaded normal distributions
    16. Winstreakdif
    17. age_dif
    18. Sig_str_dif
    19. Avg_td_dif

 



UFC 264 - Poirier VS McGregor - Results prediction

On July 10th 2021 9:00 PM CDT UFC 264 was celebrated in Las Vegas, Nevada. There was 12 programmed fights (The Yaozong VS Amedovski fight was canceled prior to the event date).  It is important toi let the reader know that these fights were not considered in the training process.

The model prediction was as accurate as in the testing. Getting 58% accuracy. The following chart describes the final results of the event.



![UFC264_results](/Images/UFC264.png)


The_ ‘Winner_predict’_ is the column with the model result predictions. The_ ‘Winner’ _column is the real result, and the _‘Result’_ column shows if the model was true or not. The following pie chart shows the percentage of right and wrong answers the model predicted.


<p align="center">
  <img src="/Images/pie2.png" />
</p>

# Conclusion

In conclusion, the model never showed a clear path to increase the prediction accuracy. More research is needed in order to define a plan of action to reach the goal of at least 80% prediction rate. All efforts mentioned in this document generated a prediction between 51% and 58%.

If the goal is possible to reach, the methods, feature combination and parameters used in this project didn’t generate evidence of its viability.


# 


# Bibliography

_Unified Rules of Mixed Martial Arts | UFC_. (n.d.). Retrieved July 7, 2021, from [https://www.ufcespanol.com/unified-rules-mixed-martial-arts](https://www.ufcespanol.com/unified-rules-mixed-martial-arts)

._UFC 264: Poirier vs McGregor 3_. (n.d.). Retrieved July 13, 2021, from [https://www.ufcespanol.com/event/ufc-264](https://www.ufcespanol.com/event/ufc-264)

_Homepage | UFC.COM_. (n.d.). Retrieved July 13, 2021, from https://www.ufcespanol.com/

_Ultimate UFC Dataset | Kaggle_. (n.d.). Retrieved July 13, 2021, from [https://www.kaggle.com/mdabbert/ultimate-ufc-dataset](https://www.kaggle.com/mdabbert/ultimate-ufc-dataset)

_Alen Amedovski vs. Yaozong Hu, UFC 264 | MMA Bout | Tapology_. (n.d.). Retrieved July 13, 2021, from [https://www.tapology.com/fightcenter/bouts/572064-ufc-264-alen-amedovski-vs-yaozong-bad-boy-hu](https://www.tapology.com/fightcenter/bouts/572064-ufc-264-alen-amedovski-vs-yaozong-bad-boy-hu)

_Artificial neural network - Wikipedia_. (n.d.). Retrieved July 13, 2021, from [https://en.wikipedia.org/wiki/Artificial_neural_network](https://en.wikipedia.org/wiki/Artificial_neural_network)

Grant, T. P. (n.d.). _History of Jiu-Jitsu: Coming to America and the Birth of the UFC_. Retrieved July 7, 2021, from [http://bleacherreport.com/articles/654500-history-of-jiu-jitsu-coming-to-america-and-the-birth-of-the-ufc](http://bleacherreport.com/articles/654500-history-of-jiu-jitsu-coming-to-america-and-the-birth-of-the-ufc)

Green, T. A., & Svinth, J. R. (2010). _Martial arts of the world : an encyclopedia of history and innovation_. 42.


# 


# Appends


## Append A

Source code of the project can be reached in the following google colab notebook...

[https://colab.research.google.com/drive/12iMl9ujySF9sVoj3womXCX1C8y5Nl11m?usp=sharing](https://colab.research.google.com/drive/12iMl9ujySF9sVoj3womXCX1C8y5Nl11m?usp=sharing)

...or in this repository.

## Append B

Ultimate UFC dataset

Public dataset available in Kaggle.

[ https://www.kaggle.com/mdabbert/ultimate-ufc-dataset](https://www.kaggle.com/mdabbert/ultimate-ufc-dataset)
