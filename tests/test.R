library(quanteda)
library(word2vec)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
lis <- unclass(toks)

type <- types(toks)
type[type %in% stopwords()] <- ""
mod <- word2vec:::w2v_train(toks, type, verbose = TRUE)
dim(as.matrix(mod))

mod2 <- word2vec:::w2v_train(unclass(toks)[1:10], types(toks), verbose = TRUE)
dim(as.matrix(mod2))

mod_lis <- word2vec(lis, dim = 50, iter = 5, min_count = 5,
                    verbose = TRUE, threads = 4)
emb_lis   <- as.matrix(mod_lis)
dim(emb_lis)
predict(mod_lis, c("people", "American"), type = "nearest")

mod_txt <- word2vec(txt, dim = 50, iter = 5, split = c("[ \n]", "\n"), min_count = 5,
                    verbose = TRUE, threads = 4)
emb_txt   <- as.matrix(mod_txt)
dim(emb_txt)
predict(mod_txt, c("people", "American"), type = "nearest")


microbenchmark::microbenchmark(
    "lis" = word2vec(lis, dim = 50, iter = 5, min_count = 5,
                     verbose = FALSE, threads = 10),
    "txt" = word2vec(txt, dim = 50, iter = 5, split = c("[ \n]", "\n"), min_count = 5,
                     verbose = FALSE, threads = 10),
    times = 10
)

