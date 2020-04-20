# Report

**Team ID:** 003

**Student Name(s):** Téo Bouvard

## Standard features and algorithms

Overall, the SVM algorithm had a better accuracy than Naive Bayes, but showed a slightly higher rate of false positives. Concerning the word weighting strategy, TF-IDF appears to be the most accurate transformer. It is hard to draw a conclusion concerning TF and Count because TF improved the accuracy when used with SVM, but significantly reduced it when combined with Naives Bayes.
The best results were attained by combining TF-IDF with the LinearSVC classifier. I can see intuitively why TF-IDF yields the best results : the more a word appears in a few documents, the more importance it should have. But I do not know enough about the inner workings of the two machine learning algorithms used to be able to explain why SVM performed better than Naive Bayes. 

#### Results

  The dataset was split with sklearn's `train_test_split` function, with a 80% training split. To be able to use this method directly, the dataset had to be organized in a specific folder hierarchy using the `organize_data.py` script. This method leads to a more homgeneous training dataset than the manual splitting implemented in assignment 1A, because the whole dataset is shuffled before splitting. This also allowed to modify the training and testing splits by simply changing the `random_state` parameter.
  All the standard results reported below were obtained without any pre-processing, by passing the wole email string to the vectorizer.


| Algorithm | Term weighting | Acc. | Prec. | FPR |
| -- | -- | -- | -- | -- |
| Naive Bayes | Count | *0.964* | *0.999* | *0.002* |
| Naive Bayes | TF | *0.913* | *1.000* | *0.000* |
| Naive Bayes | TF-IDF | *0.972* | *0.999* | *0.002* |
| SVM | Count | *0.995* | *0.996* | *0.006* |
| SVM | TF | *0.997* | *0.997* | *0.004* |
| **SVM** | **TF-IDF** | **0.997** | **0.998** | **0.003** |

The last row is the one which scored the highest accuracy on Kaggle.

## Experimental approaches

### Special character density

To continue where I stopped in assignment 1A, one of the first experimental approach I tried was extracting special character density, which is the number of occurrences of a set of special characters divided by the length of the whole email. This special density is a quick way to evaluate the number of URLs embedded in the email, the excessive use of exclamation marks, the quantity of HTML content, and other features that can help differentiate between spam and ham. I manually tested different sets of characters and the one yielding the best results was `[!#?<>$%+=()/&]`. In the same way, I tried to extract uppercase density, but this did not work at all.

### Return Path Existence

This approach is the quickest one and was inspired by a classmate during the lesson following assignment 1A, when students shared their experiences. It only checks if the email contains a return path, which is a hint that this email was probably sent to a high number of recipients, and might be a spam. This feature extraction could be improved by checking if the return path address is the same as the sender's address, or is at least a known email address, so as not to label a company mail as spam, for example.

### Sender Domain Name

This approach extracts the domain names from the sender address, and vectorizes it using CountVectorizer. The idea is that genuine emails are often sent from common domain names, like the Enron company from which the dataset seems to be collected. On the other hand, spammers tend to use exotic or shady mail services which can be identified from their domain name.

#### Text pre-processing

During my experiments, I tried to keep the text pre-processing phase basic. The only thing I did was to extract the features I wanted from the emails, some of them thanks to the email module, and feed them to the different machine learning algorithms. I did not apply any stemming nor lemmatization. The reason behind this choice is that I noticed spammers often use techniques to bypass these approaches, such as using coded language (eg. vi@gra, amb1en ...). Stemming those words would not help classification, whereas keeping them as-is helps identifying common spammer words and tricks.

#### Results

| Algorithm | Features | Other choices | Acc. | Prec. | FPR |
| -- | -- | -- | -- | -- | -- |
| SVM | Mail Length | Scaled with MinMaxScaler | 0.426 | 0.000 | 0.574 |
| SVM | Hour of Day | None | 0.574 | 1.000 | 0.574 |
| SVM | Uppercase Density | None | 0.574 | 1.000 | 0.574 |
| SVM | Special Character Density | Special character set | 0.696 | 0.640 | 0.386 |
| SVM | Return Path Existence | None | 0.946 | 0.911 | 0.108 |
| SVM | Sender Domain Name | None | 0.974 | 0.989 | 0.016 |
| SVM | Combined Features | Grid Search for hyperparameter tuning | 0.998 | 0.998 | 0.003 |

Mail Length, Hour of Day and Uppercase Density all led to a useless model. Special character density improves baseline accuracy by 12%, but may be improved further by doing a grid search to find the optimal special character set. Sender Domain Name and Return Path Existence both have a great accuracy, but still do not perform better than a simple TF-IDF of the whole email. One advantage is that they are both significantly faster than TF-IDF.

Combined Features is a FeatureUnion of TF-IDF on the whole email, Return Path Existence, Sender Domain Name and Special Character density. Despite having a slightly higher score than simple TF-IDF on my validation split, the accuracy score obtained on Kaggle was a bit lower (0.99719 vs. 0.99730).

## Discussion

The approach I used was basically the same as during assignment 1A : try different features, check what kind of errors the classifier made, tune the features for the next iteration.
I learnt about the "No free lunch" theorem, which states that there is not a single best classification method for a given dataset, and that finding one that performs well is all about trying different approaches. I also learnt about Grid Search for hyperparameter tuning, which I tried, but the best parameters found did not improve the accuracy further than the default ones, which already yielded a high score. Finally, I found out that Pieplines were really useful as they allowed to build the model in a serialized and easily understandable way.
I was as impressed by the efficiency of simple features (Return Path existence) as by the non-efficiency of features for which I had high expectations (Uppercase Density, Hour of Day).
I found it quite hard to figure out if uniting all the features I had extracted really improved the classifier, as the base accuracy obtained with TF-IDF did not allow much margin for improvement (99.7%).

## References

K, Ben. “Text Mining Preprocess and Naive Bayes Classifier (Python).” Medium, 11 June 2018, medium.com/@baemaek/text-mining-preprocess-and-naive-bayes-classifier-da0000f633b2.

scikit-learn developers. “Working With Text Data.” Scikit-Learn, scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html.

scikit-learn developers. “Sklearn.feature_extraction.Text.HashingVectorizer.” Scikit-Learn, scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html.

Joel Nothman et al. 2018. Stop Word Lists in Free Open-source Software Packages. In *Proceedings of Workshop for NLP Open Source Software*, pages 7–12

Fullwood, Michelle. “Using Pipelines and FeatureUnions in Scikit-Learn.” michelleful.github.io/code-blog/2015/06/20/pipelines/.

Wikipedia contributors. “No Free Lunch in Search and Optimization.” Wikipedia, en.wikipedia.org/wiki/No_free_lunch_in_search_and_optimization.