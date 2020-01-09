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

If we use information derived through observation (measurements, feature extraction, characterization, ...) quantified in a continuous random variable we call a feature vector x.

Our knowledge of the distribution of x for each category is expressed by the class specific density function p(x|w<sub>i</sub>) for i = 1, 2. 

IMAGE

The observaion change our certainty about which category the object belongs to. The certainty prior to observation is changed to a certainty influenced by the observation.

prior probability --> posteror probability

## Bayes rule

* p( w<sub>1</sub> | x ) = (p( x | w<sub>1</sub> ) * p( w<sub>1</sub>)) / p( x )

* p( x ) = p( w<sub>1</sub> ) * p( x | w<sub>1</sub> ) + p( w<sub>2</sub> ) * p( x | w<sub>2</sub> )

An observation of x with p( w<sub>1</sub> | x ) > p( w<sub>2</sub> | x ) will lead us to decide on w1 as it has the highest posterior probability.

Proof : The error rate for an observed x will correspond to the probabilities of the category not chosen.

p( error | x ) = p( w<sub>1</sub> | x ) if we decide w<sub>2</sub> else p( w<sub>2</sub> | x).

For a given x, we can minimize th error rate by selecting the class with the higher posterior probability.