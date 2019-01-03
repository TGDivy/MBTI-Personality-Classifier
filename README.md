# MBTI-Personality-Classifier

This project uses Natural Language processing to classify a person's personality based on their social media posts.

For additional information please checkout the presention pdf, or ppt file in the repository.

MBTI persnality classifies a person's personality into 4 broad categories, namely:
  * Extroversion and Introversion
  * Sensing and Intuition
  * Thinking and Feeling
  * Judging and Perceiving

The data was imported from Kaggle: https://www.kaggle.com/datasnaek/mbti-type

The tokenization was done using NLTK toolk, and the model was based on Bayesian Model in Natural Language processing.

For this particular problem, i trained 4 separte models each one predicted one of the attributes. The final output is based on percentage certainty the model has. 
![Image of Yaktocat](https://github.com/TGDivy/MBTI-Personality-Classifier/blob/master/Output%20Graphs%20and%20Predictions/Divy.png?raw=true)
This worked quite well in classifying my personality type, It's INTJ, although you can see that the F:J is 55:45 which is quite close. Unfortunately, some of the personality types had a lot more data, and so the model often fails to classify a personality with the lower end.

Despite that, the model has an accuracy of 80% on train:test split of 80:20 for each of the attributes.

Hopefully, in future I will try with a bigger dataset, maybe try to extract data from Reddit posts.

Here is my solution to the dataset on Kaggle: https://www.kaggle.com/tgdivy/mbti-personality-classifier

**Note:** The files have been reogranised, so you may have to change the output directories when you run the code.
