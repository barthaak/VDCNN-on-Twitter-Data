# The Code

The code used for this implementation is a small adaption of the Keras implementation for VDCNN found on https://github.com/zonetrooper32/VDCNN. 
The changes are as follows:
 - Using different learning rate (0.001) and different sequence length (140)
 - Make it usable for 3 specific datasets (see data section below)
 - Addition of potential strong dropout


# The Data

In the ‘Data’-folder there are three different type of twitter datasets. 
 - The folder ‘twitter_hate_csv’ contains a subset of the tweets used by Wasseem and Hovy (2016) in ‘Hateful symbols or hateful people? Predictive features for hate speech detection on twitter’
 - The folder ‘twitter_semeval_csv’ consists of a subset of tweets used by Nakov et al. (2013) in ‘Semeval-2013 task 2: Sentiment analysis on twitter’ 
 - Finally, the folder ‘twitter_sentiment_csv’ has a subset of tweets used by Go (2009) in ‘Sentiment classification using distant supervision’ 


# Checkpoints and logs

The ‘checkpoints’-folder and ‘logs’-folder are empty initially. 
However, while running the model the ‘checkpoints’-folder will store the weights of the model and the ‘logs’-folder will store the logs to be used by Tensorboard. 
