### The Code

The code used for this implementation is a small adaption of the Keras implementation for VDCNN (Conneau et al., 2016) found on https://github.com/zonetrooper32/VDCNN. 
The changes are as follows:
 - Using different learning rate (0.001) and different sequence length (140)
 - Make it usable for 3 specific datasets (see data section below)
 - Addition of potential strong dropout


### The Data

In the ‘Data’-folder there are three different type of twitter datasets. 
 - The folder ‘twitter_hate_csv’ contains a subset of the tweets used by Wasseem and Hovy (2016) 
 - The folder ‘twitter_semeval_csv’ consists of a subset of tweets used by Nakov et al. (2013) 
 - Finally, the folder ‘twitter_sentiment_csv’ has a subset of tweets used by Go (2009)


### Checkpoints and logs

The ‘checkpoints’-folder and ‘logs’-folder are empty initially. 
However, while running the model the ‘checkpoints’-folder will store the weights of the model and the ‘logs’-folder will store the logs to be used by Tensorboard. 


#### References:

Conneau, A., Schwenk, H., Barrault, L., & Lecun, Y. (2016). Very deep convolutional networks for text classification. arXiv preprint arXiv:1606.01781.

Go, A., Bhayani, R., & Huang, L. (2009). Twitter sentiment classification using distant supervision. CS224N Project Report, Stanford, 1(12), 2009.

Waseem, Z., & Hovy, D. (2016, June). Hateful symbols or hateful people? predictive features for hate speech detection on twitter. In Proceedings of the NAACL student research workshop (pp. 88-93).

Wilson, T., Kozareva, Z., Nakov, P., Rosenthal, S., Stoyanov, V., & Ritter, A. (2013, June). SemEval-2013 task 2: Sentiment analysis in twitter. In Proceedings of the International Workshop on Semantic Evaluation, SemEval (Vol. 13).

