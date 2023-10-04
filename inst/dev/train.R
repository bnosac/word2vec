library(word2vec)

if(require(quanteda, quietly = TRUE) && require(stringi, quietly = TRUE)){
    library(quanteda) 
    library(stringi)
    data("data_corpus_inaugural", package = "quanteda")
    corp <- data_corpus_inaugural %>% 
      corpus_reshape(to = "sentences")
    toks <- tokens(corp, remove_punct = TRUE, remove_symbols = TRUE)
    lis <- as.list(toks)
    txt <- stri_c_list(lis, " ")
    x   <- as.character(corp)
    x   <- txt_clean_word2vec(x, ascii = TRUE, alpha = TRUE, tolower = TRUE, trim = TRUE)
    lis <- strsplit(x, split = " ")
}else if(require(tokenizers.bpe, quietly = TRUE)){
    library(tokenizers.bpe)
    data(belgium_parliament, package = "tokenizers.bpe")
    x <- subset(belgium_parliament, language == "french")
    x <- x$text
    model <- bpe(x, coverage = 0.999, vocab_size = 5000, threads = 1)
    lis <- bpe_encode(model, x = x, type = "ids")
    lis <- lapply(lis, as.character)
    x   <- sapply(lis, paste, collapse = " ")
}else if(require(udpipe, quietly = TRUE)){
    library(udpipe)
    data(brussels_reviews, package = "udpipe")
    x   <- brussels_reviews$feedback
    x   <- txt_clean_word2vec(x, ascii = TRUE, alpha = TRUE, tolower = TRUE, trim = TRUE)
    lis <- strsplit(x, split = " ")
}

# list-based approach
set.seed(123456789)
mod_lis <- word2vec(lis, dim = 50, iter = 20, min_count = 3, type = "cbow", lr = 0.01)
emb_lis <- as.matrix(mod_lis)
dim(emb_lis)
# file-based approach
set.seed(123456789)
mod_txt <- word2vec(x, dim = 50, iter = 20, min_count = 3, type = "cbow", lr = 0.01)
emb_txt <- as.matrix(mod_txt)
dim(emb_txt)


## TEST equivalence
setdiff(summary(mod_lis), summary(mod_txt))
all.equal(summary(mod_lis), summary(mod_txt))
all.equal(emb_lis, emb_txt)

if(require(microbenchmark, quietly = TRUE) && FALSE){
    microbenchmark::microbenchmark(
        "lis" = word2vec(lis, dim = 50, iter = 5, min_count = 5,
                         verbose = FALSE, threads = 10),
        "txt" = word2vec(x, dim = 50, iter = 5, split = c("[ \n]", "\n"), min_count = 5,
                         verbose = FALSE, threads = 10),
        times = 10
    )
}
