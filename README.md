# Music Genre Classification

## Problem Statement
Music information retrieval is a large area of research dedicated to trying to extract useful information from music. Automatic genre classification is one of the tasks motivating this research. Automatic genre classification is useful because companies such as Spotify and Pandora can use this to recommend similar music to their users. Music databases can also use this to automatically classify the songs in their databases. Research has been done in this area, and the goal of this project is to use a convolutional neural network to classifty a song by its genre. Specifically, my metric will be accuracy, and I will compare this model to other models with commonly used features such as MFCCs to see which model performs better. 

## Notebooks
1. [Data Gathering](https://git.generalassemb.ly/danielle-mizrachi/yelp/blob/master/code/01_Yelp_Restaurant_Data_Gathering.ipynb)
2. [Data Cleaning](https://git.generalassemb.ly/danielle-mizrachi/yelp/blob/master/code/02_Yelp_Restaurant_Data_Cleaning.ipynb)
3. [Exploratory Data Analysis](https://git.generalassemb.ly/danielle-mizrachi/yelp/blob/master/code/06_EDA.ipynb)
4. [Modelling](https://git.generalassemb.ly/danielle-mizrachi/yelp/blob/master/code/07_Modeling.ipynb)
5. [Convolutional Neural Network]
6. [CNN Exploration]

## Executive Summary

### Data Gathering
The dataset I used was the GTZAN Genre Collection (found here) which was used in a well known paper on genre classification in 2002. The format of the files were .wavs, so I was able to use librosa to read them into the notebook. I wrote two functions to preprocess the data. One read in each audio file and computed several numeric features, such as MFCCs, spectral centroid, and zero-crossing rate. The other read in each audio file and computed the mel spectrogram. This is essentially an image representation of the audio file. 

### Data Cleaning
I did not have to do a lot of data cleaning. All the audio files were 30 second bits of their respective songs, so the sizes were the same. I only had to create the labels.

### Modeling with Numeric Features
The first model I built was a support vector machine with only the first 13 MFCCs (higher ones tend to be correlated). This model did ok with a training score of 71.5% and a testing score of 60.4%. I tried adding some other numeric features such as spectral rolloff, spectral centroid, and zero-crossing rate 

### Modeling with "Image" Data


## Limitations
One of the biggest limitations of this project was the size of the dataset. 100 songs per genre is a pretty small sample size for a model to be training on. In addition, genres are not easily defined. Some genres have very loose definitions -- so much so that humans argue about which genre a particular song is. Many songs also may contain elements of several genres. All in all, automatic genre classification is a difficult problem, but more data should certainly help. 


## Conclusions and Future Research
