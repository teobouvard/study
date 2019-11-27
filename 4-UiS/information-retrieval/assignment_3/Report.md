# Assignment 3 report

**Team number:** 010
**Team members:** Emile Hertoghe and Téo Bouvard

## Indexing and fields

Describe the process you used in indexing the assigned DBpedia collections, including any text pre-processing defined using Elasticsearch analyzers. Explain your reasoning behind the choices you made.

Indexing was a challenging part for multiple reasons. 

* First of all, the amount of documents we had to index is fairly important. The files we used for indexing use 77GB of disk space.
* A single document can appear multiple times in a file, but can also appear in different files. Some documents appear in every file. Therefore we have to update documents once they are already indexed.
* Elasticsearch has a very low throughput when updating indexed documents. The maximum rate of update we managed to get was ~300 documents/second. At this rate, it would already take 18 hours to index the whole collection **if** each document had to be indexed only once without being updated. If we assume the worst case where each line of each file leads to a document update, this would take 20 days because there are 529 891 072 lines in total.

To solve these challenges, we used an intermediary state between reading the files and actually indexing the documents. We created a dictionary with the the documents URI as key, and the different fields as values. At each line read, we append the object of the parsed <s, p, o> tuple to the corresponding field of the corresponding entity in the dictionary. This is much more efficient because accessing a dictionary is constant time, and appending string is faster than re-indexing the document. However, this means that at some point, the whole collection has to fit in memory. 
Once we have the whole collection as a dictionary, we can use the bulk indexing methods of Elasticsearch which allow us to have an indexing rate of approximately 10 000 documents/second. There are ~20M distinct documents in total, but we index only those having "names" and "catch_all" fields non-empty. This reduces the final number of indexed documents to ~5M. 

During indexing, we use two distinct analyzers:
* `english_analyzer` : Uses the standard tokenizer, but doesn't use the standard analyzer because it removes numbers. As numbers play an important role in search, we want to make sure they are indexed along with the other terms. That allows us to have meaningful results for queries such as `python 3`, `1984` or `8 bit`. Our custom analyzer removes the standard english stopwords, along with a few terms recurring in dbpedia literals such as `en` or `org`. Finally, this analyzer applies minimal Porter stemming.
* `whitespace` : Only tokenizes a string on whitespace, without any analyzing. This analyzer is used to index URIs for Entity Linking  + Retrieval.

List the fields that were indexed, and for each field briefly describe the rules that were applied, e.g., when the object of an SPO triple is a URI.

In the following list, "`fields`" refers to the constant dictionary defined in the 7th cell of the indexing notebook.

  - **names**: Literal objects with predicates in `fields['names']`.
  - **categories**: Resolved objects with predicates in `fields['categories']`. Resolution is performed with `category_labels_en.ttl`.
  - **similar_entity_names**: Resolved objects with predicates in `fields['similar_entity_names']`. Resolution is performed with `labels_en.ttl`. For reverse relations, subjects are treated as objects and vice versa.
  - **attributes**: Resolved predicates and literal objects with predicates matching \<dbp:*>. Resolution of predicates is performed with `infobox_property_definitions_en.ttl`.
  - **related_entity_uri**: Objects which are URIs of entities and having predicates which matches \<dbp:*>.
  - **related_entity_names**: Resolved entity URIs from the above field. Resolution is performed with `labels_en.ttl`.
  - **catch_all**: Resolved objects not fitting in any of the above fields.

All fields are indexed using `english_analyzer`, except `related_entity_uri` field which is indexed using the `whitespace` analyzer. This is because we want to identify perfect matches in the ELR model. Each field except `related_entity_uri` is indexed with term positions, so that we can compute term sequences more easily.
Once our index is built, we extract the total length of each field and write it to a file (`collection_stats.json`), so that we are able to compute background language models more easily. 

## General workflow

For each query, a first pass is performed. The first pass is done by using the `search` endpoint of Elasticsearch, with a result size of 100. Ranking is performed on these 100 documents. For each document in this first pass, a score is computed. The way this score is computed depends on the model. Once scores have been computed for each document of the first pass, they are sorted according to their score. Once all queries have been assigned a new ranking, the whole ranking is written to a file.

## Implementation of MLM

Summarize your approach in implementing MLM in `2_Ranking_Model_1.ipynb`. You may reference specific named sections of the notebook you submitted to explain your choices and the reasoning behind them.

Query terms are analyzed using the `analyze` endpoint, with the `english_analyzer`. Term vectors are retrieved using the `termvectors` endpoint, with term and field statistics enabled. Language model scores are computed for each analyzed query on `names` and `catch_all` fields. Mixture of Language Model score is computed as a linear combination of these two scores with 0.2 as weight for the `name` field and 0.8 for the `catch_all` field.

The only tricky part of the implementation is the way we retrieve the collection background model. On some documents, a query term is not present in the field being scored. We still want to be able to compute the collection background model for this term, but its total term frequency can not be retrieved from the document term vector because this specific term is not in it. The way we dealt with this problem is to use a method called `find_ttf`, which creates a new query in order to find a document with this specific term in a given field. Once we find it, we can finally get the total term frequency in this field.
One interesting thing would be to have Elasticsearch to do that internally by projecting the collection term vector on the query term vector. This would allow us to directly get all the information we need with a much more simple operation, but it doesn't exist to the best of our knowledge.

## Implementation of SDM+ELR

Summarize your approach in implementing SDM+ELR in `2_Ranking_Model_2.ipynb`. You may reference specific named sections of the notebook you submitted to explain your choices and the reasoning behind them.

The query terms feature function *f<sub>T</sub>* is computed with the same method as MLM, but only on the `catch_all` field. To compute bigram feature functions *f<sub>O</sub>* and *f<sub>U</sub>*, we use two new strategies.   

The first one is to use a method which returns a list of terms from a term vector. This allows us to easily compute the number of ordered and unordered bigram matches in a document. As with query terms, it gets tricky when we try to retrieve the bigram background collection model. Unlike the `sum_ttf` for query terms, this information can not be retrieved directly with Elasticsearch. We could retrieve all documents containing a bigram with a `match_phrase` query, and count the exact number of occurrences of this bigram after getting the termvectors for each document returned by the search. This solution is painfully slow because of the time it takes for Elasticsearch to retrieve the termvectors of a document.  

Our second strategy to solve this problem was to use an approximation of the bigram background model, which considers that the number of occurrences of a bigram in the collection is equal to the number of documents retrieved with the `match_phrase` query. While this may be relatively accurate for ordered bigrams, which rarely appear more than once in a document, this is a pretty bad approximation for unordered bigrams. The advantage of this strategy is that it is simple and efficient, because we can use nearly the same query for both types of bigrams, only setting the `slop` parameter of the query to 0 for ordered bigrams and `w` for unordered bigrams. The variable `w` determines the size of the window considered for an unordered bigram match.  

Concerning the entity feature function *f<sub>E</sub>*, we use the given entity annotations for each query and compute a score according to the ELR model. To compute the entity collection background, we once again use the `find_ttf` method, but this time on the `related_entity_uri` field. 

## Implementation of FSDM+ELR

Summarize your approach in implementing your model in `2_Ranking_Model_3.ipynb`. You may reference specific named sections of the notebook you submitted to explain your choices and the reasoning behind them.

The model we implemented is FSDM + ELR. The implementation is nearly identical to SDM + ELR, except that we compute scores on each field and not only on the `catch_all` field. Scores are computed with the same methods as with SDM, sequentially accumulated on each field to get the total field scores.

## Tricky part with all models

One recurring problem is that sometimes a term, a bigram or an entity might not appear even once in the whole collection. This leads to a feature function equal to zero, but scores are computed in log space. The first strategy used was to just skip these terms, but that would have the same effect as computing a feature function of 1, which is a good score. The final strategy used is to inflict a penalty to these terms by assigning a fixed feature score defined by the constant `PENALTY`. This constant was approximately tuned to 10<sup>-6</sup> and offered an improvement of ~2% NDCG score.

## Results

  - *Report the performance results in the table below, using each of your three models to output entity ranking predictions given `queries.txt`. Evaluate the models' predictions with respect to `qrels.csv` using `3_Evaluation.ipynb`.*

| Model | NDCG@10 | NDCG@100 |
| -- | -- | -- |
| First pass (BM25) | 0.3435 | 0.3857 |
| MLM title+content | 0.3410 |0.4114 |
| SDM+ELR | 0.3190 | 0.3987 |
| FSDM+ELR | 0.3424 | 0.4132 |

## Discussion

  - *Discuss the work done, the results, and the understanding gained from this experience. Outline the patterns you observed relating to the models' mechanisms and their respective performance.*
  - *Describe alternative approaches you tried in developing your own model, if any.*
  - *What are other approaches that could be useful, which you didn't try? Make an argument why these hypothetical approaches might be effective.*

We manually implemented different scoring models and evaluated them against query relevance judgments. More complex models such as FSDM + ELR seems to be yielding better results than SDM + ELR, but not by an incredible margin. However, manual computation of these more complex models is a much slower, with a response time increasing by a factor of ~10. Considering user experience, this delay is non negligible and should be taken into account when choosing a retrieval model.  
We also tried the different scoring models offered directly by Elasticsearch. Because they are designed to use the native data structures used internally by Elasticsearch they are a lot faster, and some of them perform better than all our models. Divergence From Independence with a standardized independence measure is the model which yielded the highest NDCG score. The performance of all models tried for the first pass is listed below.

| Model | NDCG@10 | NDCG@100 |
| -- | -- | -- |
| BM25 | 0.3435 | 0.3857 |
| LM + Dirichlet smoothing | 0.3179 | 0.3668 |
| LM + Jelinek Mercer smoothing | 0.3383 | 0.4021 |
| DFI standardized | 0.3791 | 0.4421 |
| DFI saturated | 0.3374 | 0.4040 |
| DFI chi-squared | 0.3749 | 0.4388 |

We did not try to incorporate Word Embeddings because the indexing part took already quite a bit of time, but we are confident that doing so would increase our model performance. We think that word embeddings can more naturally capture the intent of a query than a combination of n-grams matches.

## References

  - Elasticsearch Reference [online]
