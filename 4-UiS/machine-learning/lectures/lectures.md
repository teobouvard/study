## What ML is about

The focus is on learning from data by exploiting model structures and patterns that can be used to make predictions on future generations of data from the same underlying process. Many disciplines focusing on these matters : statistical learning, pattern recognition, signal and image processing, computer science, data mining ...

## Classification

The case of discriminating salmon from sea bass is an exercise of classification problems.

![Classification example](img/classification.png)

## Regression

Continuous version of classification.

## Formalizing the introductory problem

class 1 : sea bass (w1)
class 2 : salmon (w2)

Aim is to discriminate between objects of the two classes with the minimum error rate.


P(w1) = prior probability that the observed fish is a sea bass
P(w2) = prior probability that the observed fish is a salmon

Only knowing prior probabilities (without observation), the optimal decision rule would be to classify samples as the largest prior probability, ie. choosing w1 if P(w1) > P(w2) else w2 leading to an error of w2.