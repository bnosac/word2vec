library(quanteda)
library(word2vec)

data_corpus_guardian <- readRDS("/home/kohei/Dropbox/Public/data_corpus_guardian2016-10k.rds")
corp <- data_corpus_guardian %>% 
#corp <- data_corpus_inaugural %>% 
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE) %>% 
    tokens_remove(stopwords(), padding = TRUE) %>% 
    tokens_tolower()
ndoc(toks)

mod <- word2vec:::w2v_train(toks, types(toks), verbose = TRUE, size = 300, 
                             iterations = 5, minWordFreq = 5, threads = 6)
dim(as.matrix(mod))
predict(mod, c("people", "american"), type = "nearest")

require(LSX)
lss <- as.textmodel_lss(t(as.matrix(mod)), "good")
head(coef(lss))
tail(coef(lss))

lis <- as.list(toks)
mod_lis <- word2vec(lis, dim = 50, iter = 5, min_count = 5,
                    verbose = TRUE, threads = 4)
emb_lis   <- as.matrix(mod_lis)
dim(emb_lis)
pred_lis <- predict(mod_lis, c("people", "American"), type = "nearest")

#saveRDS(mod_lis, "tests/word2vec_v04.RDS")

microbenchmark::microbenchmark(
    "lis" = word2vec(lis, dim = 50, iter = 5, min_count = 5,
                     verbose = FALSE, threads = 10),
    "txt" = word2vec(txt, dim = 50, iter = 5, split = c("[ \n]", "\n"), min_count = 5,
                     verbose = FALSE, threads = 10),
    times = 10
)

