library(quanteda)
library(word2vec)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
lis <- as.list(toks)
txt <- stringi::stri_c_list(lis, " ")

mod_lis <- word2vec(lis, dim = 15, iter = 20, min_count = 2,
                        verbose = FALSE, threads = 1)
#emb_lis   <- as.matrix(mod_lis)

mod_txt <- word2vec(txt, dim = 15, iter = 20, split = c("[ \n]", "\n"), min_count = 2,
                    verbose = FALSE, threads = 1)
emb_txt   <- as.matrix(mod_txt)
predict(mod_txt, c("citizen", "country"), type = "nearest")

n <- 100
thread <- 3
id <- 1
floor((n / thread) * id)
floor((n / thread) * (id + 1)) - 1
