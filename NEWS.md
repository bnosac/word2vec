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
