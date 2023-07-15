# Tackling Misinformation through Tweets: A Comparative Study of Various Machine Learning Approaches

## Abstract
Fake news or deceptive journalism is one of the most researched and worked-on projects in the machine learning world. In our modern and expeditious world, getting information and knowledge is easier to access on the internet than other sources. Social media, websites, and blogs have completely replaced newspapers and articles for the majority of the population. This in fact has caused the rise of more destructive issues. Newspaper agencies always have a verified and trustworthy news source, but with the rise of internet usage and easy access, fake news is at an all-time high. Numerous studies have been conducted
that document the effect of misleading and inaccuracy information on the general public and how it has affected the daily commute and working of the common people. Fake news is basically fabricated posts with fabricated evidence with visual content that is completely edited using the best software available. This study employs a number of machine learning approaches to address this issue. The research was conducted on the Fake and Real News dataset, tackling misinformation through tweets. Several machine learning algorithms like Logistic Regression, Gaussian Naive Bayes, Passive-Aggressive Classifier, and LSTM machine learning models were used to tackle this challenge. The LSTM model, with an accuracy of 99.23% came out as the best model for segregating real news from fake news.

**Keywords** Deep Learning · Fake News · Gaussian Naive Bayes · Logistic Regression · LSTM · Neural Networks · Passive-Aggressive Classifier

## Introduction
News has always been so important because it informs the public about all the events and occurrences happening around the globe, as well as how they are
affecting them as a general citizen. Getting briefed on the ongoing happenings around the globe is a necessary need. There are numerous options for getting
familiar with the news around us, there are newspapers, articles, online news websites, blogs, vlogs, and social media applications [2].
It can directly or indirectly affect a person; thus, it is a very sensitive topic.

However, there are also individuals that disseminate false information, which can cause a person a lot of issues. The work of forged news or misinformation
is litigable and could be classified as a heinous crime. It has been around even before the existence of the internet. There could be global outbreaks and mishaps with the floating of fake news, and determining the existence of this fake news is of the utmost priority.

In the modern era, there’s been a huge rise in the web and the internet. And nowadays nearly everyone has access to the web, as it’s so cheap and widespread
across the world.
The widespread use of social media platforms and the internet over the past 10 years has been a very easy way to increase the abundance of fake news [6]. With a finger click, anyone can access the internet, and within seconds they can access any news article or blog, which is a very mind-boggling reality. Along with these luxuries come the issues of falsified news articles, which can cause living horrors for the people accessing these sites. People around the globe can now easily access and follow the events of personal interest as well as obstacles arising globally. They can stay up to date with everything happening around them, even in the farthest corner of the world. And this process of easy access to information has become easier with the rise of portable devices such as mobile phones and laptops.

But, as much as this has been helpful for everyone to get the news delivered fast and with a single button click, the internet has caused numerous problems
in the form of “misinformation”, “misleading”, fraud,” or “fake” news.

Misinformation can, or is, easily floated around the internet all the time because all it takes is one source to widespread ”fake” news around the internet.
And people around the world see the misinformation and can share and pass it on to others, without checking the source or confirming the actual legitimacy
of the information. This leads to several unwanted complications and problems around the globe.

Having a prevention system for this misleading and falsified information is essential and ethical. There are various legal recourses for these issues, like the International Fact-Checking Network (IFCN) or other manually checked truthdetermining websites like Snopes, Truth or Fiction, Fact-Check, or the Washington Post, which are working vastly to rectify and verify different information around the internet and pinpoint the delusional and unethical news that might cause an uproar in the society [10].

Most experts and international researchers claim that artificial intelligence, or AI, can assist in the fight against fake news. As hardware has become progressively cheaper, larger datasets are starting to become available, AI systems have recently improved their performance on a range of classification tasks and have kept improving their models to fit in more data, which in fact provides more accuracy in the models (voice detection, picture recognition, etc.).

Websites may appear reliable and effective, but they have scalability problems when dealing with massive amounts of data. The automatic fact-checking concept, which consists of three elements, was developed to overcome this problem [11]. The steps in the identification, verification, and correction process are recognition, validation, and correction. Together, the three elements debunk false claims, confirm their veracity, and disseminate updated, accurate information on social media networks.

**Problem Statement** This project is concerned with recognizing a solution that could be used to detect and filter out fake news for the purpose of helping
users to rule out misleading information. By wisely utilizing different machine learning, natural language processing, and neural network techniques, a model can be formed for detecting misleading information.

## Literature Review
An abundant number of scholars have works that show that the issue of fake news floating around the internet has been a major concern for scholars from different backgrounds. The research shows that misleading or misinformation information is now a threat to information security.

The two main categories of false news detection methodologies are as follows: (a) manual fact-checking, and (b) AI-based detection [2]

Many fact-checking websites use human opinion to determine if the news is accurate. However, it is incredibly difficult and extremely slow to manually classify news on the internet [2].

Artificial intelligence models can be used to categorize the news as fake or true and the sources as trustworthy or not, which is a less-than-ideal but still a very scalable solution.

Context-based methods can be used as they are able to understand the human context and take a human-centric view as an approach to AI. These contextbased methods use stylometric and linguistic features for classification. Sentence length, frequency, and other stylistic characteristics are some of the stylometric traits that can be used. Sentence segmentation, tokenization, and POS tagging are some of the stylometric techniques that can be used. And linguistic features such as bag-of-words, frequency of words, and lexical features are used to classify a phoneme or word [4].

A paper regarding a similar topic was published by the students of Nanyang Technological University, Singapore. They had a good run on the workings of a
supervised learning model using the 2292 BuzzFeed news articles related to the elections that were held in 2016. Their results showed that only about 60% of
the accuracy was being classified correctly [16].

Girgis et al. recommended GRU models and LSTMs, which are comparable to standard RNN models, as classification models for identifying bogus news. They
used the LIAR dataset, which contains 12,836 condensed utterances broken down into different context groups. During compilation, they separated each phrase
and omitted phrases that were unnecessary. Through their deep insight and proper modeling, they compared the performances, accuracies, and results of
these models, and they were astounded to see that the overall performance of GRU was overtaking that of LSTM and Vanilla [4].

The contemporary methodology proposed by Karini et al. for determining fake news using the structural concepts of LSTM and BiGRNN had various
stout results. The authors tested their approach using a variety of datasets. In addition, they used machine learning techniques like SVM along with various
deep learning techniques like LSTM and a hybrid (BiGRNN) for classification was utilized which produced an accuracy of 82.19% [8].

Through thorough research and numerous previous works of different models working on this topic, we found that there were different problems for each
model, and accordingly decided to work on a separate one.

| References | Dataset | Contributions | Accuracy achieved | Drawbacks |
| :--------- | :-----: | :------------ | :---------------: | :-------- |
| [14]       | Extracted using FakeNewsNet tool | LSTM, RNN, and GRU deep learning techniques was utilized to research binary classification | 45%-GRU, 62%- RNN, 75%-LSTM | It doesn’t achieve greater precision and is restricted to a small dataset |
| [13]       | TwitterBR, FakeOrReal News, FakeNewsData1, btv lifestyle | Contextual features, consumer features, geometrical characteristics, mindset features, and projected features were considered as features for a classifier. | 79% | It is not possible to further improve accuracy. |
| [12] | 2282 news articles on the US election | There have been some new additions to the training classifiers. | 85% | Only done on a small dataset. The dataset can work on a specific type of news.|
| [7] | BuzzFeed dataset and PolitiFact dataset | XGBoost and DNN are implemented to classify the retrieved features | 85.86% | The extracted features using content and context was not demonstrated.|
| [1] | 5800 Twitter tweets | BiLSTM-CNN can determine the legitimacy of tweets by confirming them with specific training features. | 86.12% | Performed on a very small dataset. The result was only able to classify whether something is false or not, you can only use text-based features.|
| [9] | Twitter dataset, Weibo dataset | Appropriate and unlabeled samples are used to train CNN. Model results were verified using five-fold cross-validation. | 90% | The dataset was very small. The Twitter dataset had only 1,111 entries and the Weibo datasethad 816 entries.|
| [15] | Italian Facebook dataset | Contextual features, consumer features, geometrical characteristics, mindset features, and projected features were considered as features for a classifier. | 91% | Unable to spot the factor that has detrimental effect on the data.|
| [5] | Liar dataset | DSSM-LSTM classifies news utilizing semantic features to identify disinformation. | 99% | Feature extractions based on semantic characteristics were not shown.|

## Proposed Methodology
This section outlines the classification approach. This idea leads to the development of a method for detecting fake articles. Using supervised machine learning, the dataset is categorized in this method. The first step in this extensive fake and real tweets classification process is to collect the dataset, the data is then preprocessed, and the features are extracted from it. After that, the training and evaluation of the classifier models take place. The technique includes various experiments on a dataset that make use of LSTM [11], Logistic Regression, Passive-Aggressive Classifier, and Naive Bayes Classifier for segregating fake news from factual information. Trials are done independently on each algorithm, and the research aims to get as much accuracy as possible. 

The main goal is to use a variety of classification techniques to build a classification model that can accurately recognize news and be integrated into a Python application to search for false news data. In order to deliver a well-performing model, the Python code has also undergone the necessary refactorings. Several algorithms are applied to the data in order to detect bogus news. Various metrics (like accuracy, recall, precision, f1-score, etc.) of the results were obtained and reviewed in order to get the desired result. Here, a detailed explanation of each sub-step is given:

###  Dataset
The first step was gathering a dataset of fake news. This study used Kaggle’s Fake and Real News Dataset. It consisted of a collection of tweets ranging from 2015 to 2018.

### Data Preprocessing
Firstly, the two datasets— fake and real news datasets were combined. There was a bit of an imbalance, and thus, 20,000 random samples were selected from both
of the datasets. These samples were then pooled to create a new dataset, which was eventually utilized to train the model. The data consisted of various subjects, and they were specified with the data. Fig. 1. shows the subject distribution of the data.
