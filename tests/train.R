library(quanteda)
library(word2vec)

corp <- data_corpus_inaugural %>% 
    corpus_reshape()
toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
lis <- as.list(toks)
txt <- stringi::stri_c_list(lis, " ")

mod_txt <- word2vec(txt, dim = 15, iter = 20, split = c(" ", ".\n?!"))
emb_txt   <- as.matrix(mod_txt)

mod_lis <- word2vec(lis, dim = 15, iter = 20)
emb_lis   <- as.matrix(mod_lis)


predict(mod_txt, c("citizen", "country"), type = "nearest")
nn


