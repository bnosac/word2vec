## CHANGES IN word2vec VERSION 0.2

- Extended predict.w2v with nearest if you pass on a vector or matrix. This allows to perform word2vec analogies or extract other similarities.
- Added word2vec_similarity
- Change classes returned by word2vec to 'word2vec_trained' and read.word2vec to 'word2vec'
- Add detailed docs of predict.word2vec and as.matrix.word2vec
- Added normalize option in read.word2vec usefull when wanting to extract the raw embedding (e.g. trained with other software)
- Use Rcpp::runif as initialiser of embeddings instead of std::mt19937_64

## CHANGES IN word2vec VERSION 0.1.0

- Initial package based on https://github.com/maxoodf/word2vec commit ad08b14ba6b554a10284c59c473ee81cb7f3af34
