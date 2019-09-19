# Report

**Team ID:** 003

**Student Name(s):** Téo Bouvard

## Approach

This classifier is based on the extraction of meaningful features in emails, which are used to train a model.

### Training

Training the model consists of counting the occurences of each feature, and assigning the resulting counters to the model's attributes. The features identified as meaningful are :

- words in the whole email (first model iteration)
- words in the subject (second model iteration)
- domain names from the sender's address (third model iteration)
- special characters (aborted iteration)

Each feature is extracted from emails via regular expressions<sup>[[1]]</sup>, and occurences are tracked by Counter objects<sup>[[2]]</sup>. These particular features were chosen by manually analyzing some of the wrongly classified emails after each iteration, and identifying features that could have helped the classifier in making a better decision.

### Predicting

Once the model is trained, it can make predictions about incoming emails. Features occurences are counted in the same way as during training. Email counters are then compared with spam and ham counters from the model. This similarity comparison is done by computing the intersecton between the different counters. I don't think this is the most powerful way to compute class similarity, but it is pretty efficient as the intersection operation is natively supported by Counter objects, and it yields correct results. 
The values of the resulting similarity counters are then summed to compute a feature score. Finally, a linear combination of the features scores is computed for each class, determining a spam score and a ham score.

The predicted class is the one having the highest score.

## Experimental setup

During development, I used a simple train/validation split strategy. A specific list of filenames was created by splicing the full datafile list depending on the mode used (`--train` or `--predict`). Later, I implemented k-fold cross-validation in order to make sure I was not just trying to fit the validation split. This approach yielded more realistic metrics but significantly increased model training time, as it was done by sequentially training a new model for each fold. Both strategies can still be tested using the `--strategy ['none', 'split', 'cross']` parameter.

## Results

These results are obtained with a 70/30 split strategy with Counters of size 5000. Incrementing the maximum size of the Counters improves Accuracy but slows down model training.  
  
| Model (Chosen features) | Accuracy | Precision | FPR |
| -- | -- | -- | -- |
| Baseline | 0.602 | 0.602 | 1.000 |
| Domain | 0.557 | 0.994 | 0.002 |
| Subject | 0.611 | 0.918 | 0.053 |
| Words | 0.936 | 0.930 | 0.111 |
| Words+Subject | 0.930 | 0.954 | 0.068 |
| **Words+Domain** | **0.952** | 0.956 | 0.068 |
| Words+Domain+Subject | 0.946 | 0.971 | 0.042 |

Some models, despite being not very accurate, have a very low false positive rate (eg. Domain, Subject). By trial and error, I found that combining these models with one having a correct accuracy (eg. Words) improved the metrics. I think that a lot of other features could be added, like the density of special characters, or even the hours at which a mail is sent, but I wanted to keep models simple.
Furthermore, I think the way I compute weights to normalize each feature is a bit made up. The tf-idf way makes a lot more sense, but I did not know about it at the time, and came up with something to implement this normalization.

## Discussion

This approach was derived incrementally from error analyis after each evaluation of a new model. I saved False Positives and False Negatives filenames to an output file, and manually reviewed some of them to check if I found obvious features which could have helped the model make a better prediction. This process worked correctly, but I felt like I could easily lose a lot of time by trying to fine-tune each model (eg. fine-tuning regular expressions, counter size, or score calculation).

When I realized only one CPU core was maxed out during the (quite long) model training and I decided to parallelize this task, I learnt about the limitations of CPython when it comes to multithreading because of its Global Interpreter Lock, and discovered that there were different implementations of Python. I ended up not multithreading the script, but it would definitely be needed to speed up the training. 

I wondered way too long about an efficient way to serialize the model so as to implement saving and loading methods. I started to compare JSON and YAML performance before realizing the models did not need to be human-readable. I ended up dumping the whole class into a pickle, which I think is the most efficient. This made me learn about model sharing, and the security concerns about unknown pickles<sup>[[3]]</sup>.

## Final note

>Some people, when confronted with a problem, think "I know, I’ll use regular expressions." Now they have two problems.
>
> <cite> &mdash; Jamie Zawinski <sup>[[4]]</sup></cite>

## References

[1]:https://docs.python.org/3/library/re.html
<sup>[[1]]</sup> “Re - Regular Expression Operations.” Re - Regular Expression Operations - Python 3.7.4 Documentation.

[2]:https://docs.python.org/3/library/collections.html#collections.Counter
<sup>[[2]]</sup> “Collections - Container Datatypes.” Collections - Container Datatypes - Python 3.7.4 Documentation.

[3]:https://hackernoon.com/dont-trust-a-pickle-a77cb4c9e0e
<sup>[[3]]</sup> Lawnboy, Max. “Don't Trust a Pickle.” Hackernoon.

[4]:http://regex.info/blog/2006-09-15/247
<sup>[[4]]</sup> Friedl, Jeffrey. “Source of the Famous ‘Now You Have Two Problems’ Quote.” Jeffrey Friedl's Blog, 15 Sept. 2006.


