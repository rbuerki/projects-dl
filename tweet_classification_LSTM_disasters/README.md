# Real or Not? NLP with Disaster Tweets

A repository containing code for sentiment analysis. The goal of the model is to predict which Tweets are about real disasters and which ones are not. The goal of this project was to figure out how I can combine a self-trained word2vec embedding with a sentimant analysis LSTM. (Knowing that using available pre-trained embeddings like FastText or GloVe would probably yield better results.)

Altough I was able to load the learned embeddings into the embedding layer of the LSTM, THERE IS A BUG IN THE DATA FORMAT I CANNOT SOLVE / LOCATE. So this project is not completed!

By the way.The `EDA` Notebook is really nice and worth looking at.

### Acknowledgments
This dataset was created by the company figure-eight and originally shared on their [‘Data For Everyone’ website](https://www.figure-eight.com/data-for-everyone/).

This project is part of Kaggle's 'Getting Started' prediction competitions, see [here](https://www.kaggle.com/c/nlp-getting-started/overview).