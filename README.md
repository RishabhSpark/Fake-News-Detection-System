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

_Table 1: Literature Survey_
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

![Fig. 1.](https://github.com/RishabhSpark/Fake-News-Detection-System/blob/main/Fake-News-Detection-main/charts%20and%20graphs/combined_data_distribution.eps)

During the preprocessing of the data, we performed rough noise removal from the data first. Then, using the NLTK Python library, POS was performed and was then used to select the desired attributes or features

### Model Creation
80% of the data from both the fake and real datasets are randomly selected to be used in full dataset, leaving the remaining 20%, or 8000 samples, to be used as a testing set after the model is complete. Before a classifier can be applied to text data, it must first be preprocessed, so it tokenizes words and
uses several NLP techniques to handle POS data to reduce noise. In order for machine learning algorithms to use the generated data as input, it must be
encoded as integers and floating-point numbers. The research was performed on various tokenization, feature extraction, and vectorization techniques like Count Vectorizer, Tf-Idf Vectorizer, and Word2Vec [3, 12].

After that, the dataset is partitioned and different ML techniques like Logistic Regression, Naive-Bayes Classifier, Passive-Aggressive Classifier, and LSTM
models are used to generate the proposed classifier model. The dataset is successfully preprocessed by the algorithm before the trained part is subjected to
algorithmic application. The model is constructed with a response message after applying each model. After getting the results from the training data, the model is evaluated on the test data for further acceptability and verification. This completed the model creation part. Then the model is sent out to perform on and face the user-selected unknown data and perform predictions for the same. Also, a graph depicting the layers and dimensions of the deep learning LSTM model is shown below (in Fig. 2.).

![Fig. 2.](https://github.com/RishabhSpark/Fake-News-Detection-System/blob/main/Fake-News-Detection-main/charts%20and%20graphs/lstm_model.eps)

## Results
After evaluating the model on different parameters and scales, the study showcased that LSTM models outperformed logistic regression, passive-aggressive classifiers, and the Gaussian Naive Bayes model for the given problem by a significant margin.

The accuracy achieved by the LSTM model was 99.23%. On the other hand, logistic regression and passive-aggressive classifier were not that far behind, with
an accuracy of 94.95% and 93.4% respectively. Gaussian Naive Bayes classifier was relatively weak, with an accuracy of only 88.3%.

_Table 2. Classification Report of different models on test data_
| Model |      | Precision | Recall | F1 Score | Support |
| :---- | :--: | :-------- | :----: | :------- | :-----: |
|Logistic Regression | Fake | 0.96 | 0.93 | 0.95 | 3752 |
| | True | 0.94 | 0.97 | 0.95 | 4248 |
| | Accuracy  | | | 0.95 | 8000 |
| Gaussian Naive Bayes | Fake |  0.92 | 0.83 |0.87 | 3752 |
| | True | 0.86 | 0.93 | 0.89 | 4248 |
| | Accuracy  | | | 0.88 | 8000 |
| Passive-Agressive Classifier | Fake | 0.93 | 0.93 |0.93 | 3752 |
| | True | 0.94 | 0.94 | 0.94 | 4248 |
| | Accuracy  | | | 0.93 | 8000 |
| LSTM | Fake | 0.99 | 0.99 |0.99 | 3752 |
| | True | 0.99 | 0.99 | 0.99 | 4248 |
| | Accuracy  | | | 0.99 | 8000 |

_Table 3. Accuracy of different models_
| S. No. | Model | Accuracy |
| :----- | :---: | :------- |
| 1. | Logistic Regression | 94.95% |
| 2. | Gaussian Naive Bayes | 88.3% |
| 3. | Passive-Aggressive Classifier | 93.4% |
| 4. | LSTM | 99.23% |

## Conclusion
The findings from this research show that LSTM-based models have a great deal of potential for enhancing the precision and effectiveness of fake news identification systems. The creation of efficient technologies for identifying and battling fake news is essential as the issue of disinformation continues to gain relevance. However, there is still much potential for improvement and model refining, as the data currently available is not sufficient for this model to be fully trusted.

## Future Work
As mentioned earlier, there is a lot of room for improvement, and to begin with, scalability and generalization are one of the main drawbacks. The research focuses on a specific dataset for a specific time period. To enhance the applicability of the proposed approach in real-world scenarios, the model needs to be tested on a more extensive dataset. The proposed model needs to be tested on a diverse dataset with news from different domains, including politics, finance, sports, and entertainment, to verify its generalization ability. An extension of the model can be proposed based on the time period of certain events and such. Additionally, the research could explore methods to scale up the proposed approach, and maybe increase the computation power for real-time fake news detection on a large scale, considering the speed and accuracy of the model.

Secondly, transfer learning can be used. Transfer learning is a technique used to reuse pre-trained models and fine-tune them on new datasets. In this case,
pre-trained models such as BERT or GPT can be used to detect fake news. The model can be further trained on larger datasets, and fine-tuned for research
purposes regarding misinformation detection. The proposed approach can be compared with transfer learning techniques to evaluate its performance and to
determine which method yields better results.

Also, maybe a new model with fuzzy logic can be applied to certain aspects of fake news detection, not classification, but mostly for measuring the degree
of uncertainty or ambiguity in a tweet or a news article, as a tweet or an article can be semi-accurate [13].

## References
1. Asghar, M.Z., Habib, A., Habib, A., Khan, A., Ali, R., Khattak, A.: Exploring deep neural networks for rumor detection. Journal of Ambient Intelligence and Humanized Computing 12, 4315–4333 (2021)
2. De Beer, D., Matthee, M.: Approaches to identify fake news: a systematic literature review. Integrated Science in Digital Age 2020 pp. 13–22 (2021)
3. Faustini, P.H.A., Covoes, T.F.: Fake news detection in multiple platforms and languages. Expert Systems with Applications 158, 113503 (2020)
4. Girgis, S., Amer, E., Gadallah, M.: Deep learning algorithms for detecting fake news in online text. In: 2018 13th international conference on computer engineering and systems (ICCES). pp. 93–97. IEEE (2018)
5. Jadhav, S.S., Thepade, S.D.: Fake news identification and classification using dssm and improved recurrent neural network classifier. Applied Artificial Intelligence 33(12), 1058–1068 (2019)
6. Jose, X., Kumar, S.M., Chandran, P.: Characterization, classification and detection of fake news in online social media networks. In: 2021 IEEE Mysore Sub Section International Conference (MysuruCon). pp. 759–765. IEEE (2021)
7. Kaliyar, R.K., Goswami, A., Narang, P.: Deepfake: improving fake news detection using tensor decomposition-based deep neural network. The Journal of Supercomputing 77, 1015–1037 (2021)
8. Karimi, H., Tang, J.: Learning hierarchical discourse-level structure for fake news detection. arXiv preprint arXiv:1903.07389 (2019)
9. Liu, Y., Wu, Y.F.B.: Fned: a deep network for fake news early detection on social media. ACM Transactions on Information Systems (TOIS) 38(3), 1–33 (2020)
10. Pavleska, T., Skolkay, A., Zankova, B., Ribeiro, N., Bechmann, A.: Performance ˇ analysis of fact-checking organizations and initiatives in europe: a critical overview of online platforms fighting fake news. Social media and convergence 29, 1–28 (2018)
11. Rajalaxmi, R., Narasimha Prasad, L., Janakiramaiah, B., Pavankumar, C., Neelima, N., Sathishkumar, V.: Optimizing hyperparameters and performance analysis of lstm model in detecting fake news on social media. Transactions on Asian and Low-Resource Language Information Processing (2022)
12. Reis, J.C., Correia, A., Murai, F., Veloso, A., Benevenuto, F.: Supervised learning for fake news detection. IEEE Intelligent Systems 34(2), 76–81 (2019)
13. Satpathy, S., Prakash, M., Debbarma, S., Sengupta, A.S., Bhattacaryya, B.K.: Design a fpga, fuzzy based, insolent method for prediction of multi-diseases in rural area. Journal of Intelligent & Fuzzy Systems 37(5), 7039–7046 (2019)
14. Vereshchaka, A., Cosimini, S., Dong, W.: Analyzing and distinguishing fake and real news to mitigate the problem of disinformation. Computational and Mathematical Organization Theory 26, 350–364 (2020)
15. Vicario, M.D., Quattrociocchi, W., Scala, A., Zollo, F.: Polarization and fake news: Early warning of potential misinformation targets. ACM Transactions on the Web (TWEB) 13(2), 1–22 (2019)
16. Zhao, X., Yu, H., Li, S., Zhang, J.: A review of the research on the influencing factors of internet user information dissemination. In: 2022 5th International Conference on Pattern Recognition and Artificial Intelligence (PRAI). pp. 1311–1317. IEEE (2022)
