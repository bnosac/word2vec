## CHANGES IN word2vec VERSION 0.4.0

- Drop C++11 specification in Makevars
- Building a word2vec model is now possible by providing a list of tokenised sentences (issue #14)
  - word2vec is now a generic function with 2 implemented methods: word2vec.character and word2vec.list
  - The embeddings with the file-based (word2vec.character) and list-based approach (word2vec.list) are proven to be the same if the tokenisation is the same and the hyperparameters of the model are the same
  - In order to make sure the embeddings are the same the vocabulary had to be sorted according to the number of times it appears in the corpus as well as the token itself in case the number of times the 2 tokens occur is the same. This has as a consequence that the embeddings generated with version 0.4.0 will be slightly different as the ones obtained with package version < 0.4.0 due to a possible ordering difference in the vocabulary
  - examples provided in the help of ?word2vec and in the README
- writing text data to files before training for the file-based approach (word2vec.character) now uses useBytes = TRUE (see issue #7)

## CHANGES IN word2vec VERSION 0.3.4

- Remove LazyData from DESCRIPTION as there is no data to be lazy about
- Add option type to word2vec_similarity to allow both 'dot' similarity which is the default as 'cosine' similarity (requested in issue #5)

## CHANGES IN word2vec VERSION 0.3.3

- Allow doc2vec also to be used on word2vec_trained
- Add txt_clean_word2vec

## CHANGES IN word2vec VERSION 0.3.2

- Make example conditionally on availability of udpipe

## CHANGES IN word2vec VERSION 0.3.1

- word2vec gains argument encoding

## CHANGES IN word2vec VERSION 0.3.0

- Add doc2vec

## CHANGES IN word2vec VERSION 0.2.1

- Fix R CMD check warning message on Fedora clang

## CHANGES IN word2vec VERSION 0.2

- Extended predict.w2v with nearest if you pass on a vector or matrix. This allows to perform word2vec analogies or extract other similarities.
- Added word2vec_similarity
- Change classes returned by word2vec to 'word2vec_trained' and read.word2vec to 'word2vec'
- Add detailed docs of predict.word2vec and as.matrix.word2vec
- Added normalize option in read.word2vec usefull when wanting to extract the raw embedding (e.g. trained with other software)
- By default models trained with version 0.2 of this R package do normalization upfront before saving the model. For version 0.1 of this package this was not the case so load these in with option normalize set to TRUE
- Use Rcpp::runif as initialiser of embeddings instead of std::mt19937_64
- Functionalities default usage assumes UTF-8 encoding and predict.w2v now returns character text instead of factors
- Added read.wordvectors

## CHANGES IN word2vec VERSION 0.1.0

- Initial package based on https://github.com/maxoodf/word2vec commit ad08b14ba6b554a10284c59c473ee81cb7f3af34
