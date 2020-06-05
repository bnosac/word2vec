#' @title Train a word2vec model on text
#' @description Construct a word2vec model on text. The algorithm is explained at \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @param x a character vector with text or the path to the file on disk containing training data
#' @param type the type of algorithm to use, either 'cbow' or 'skip-gram'. Defaults to 'cbow'
#' @param dim dimension of the word vectors. Defaults to 50.
#' @param iter number of training iterations. Defaults to 5.
#' @param lr initial learning rate also known as alpha. Defaults to 0.05
#' @param window skip length between words. Defaults to 5.
#' @param hs logical indicating to use hierarchical softmax instead of negative sampling. Defaults to FALSE indicating to do negative sampling.
#' @param negative integer with the number of negative samples. Only used in case hs is set to FALSE
#' @param sample threshold for occurrence of words. Defaults to 0.001
#' @param min_count integer indicating the number of time a word should occur to be considered as part of the training vocabulary. Defaults to 5.
#' @param split a character vector of length 2 where the first element indicates how to split words and the second element indicates how to split sentences in \code{x}
#' @param stopwords a character vector of stopwords to exclude from training 
#' @param threads number of CPU threads to use. Defaults to 1.
#' @return an object of class \code{w2v_trained} which is a list with elements 
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{data: a list with elements file: the training data used, stopwords: the character vector of stopwords, n}
#' \item{vocabulary: the number of words in the vocabulary}
#' \item{success: logical indicating if training succeeded}
#' \item{error_log: the error log in case training failed}
#' \item{control: as list of the training arguments used, namely min_count, dim, window, iter, lr, skipgram, hs, negative, sample, split_words, split_sents, expTableSize and expValueMax}
#' }
#' @references \url{https://github.com/maxoodf/word2vec}
#' @seealso \code{\link{predict.word2vec}}, \code{\link{as.matrix.word2vec}}
#' @export
#' @examples
#' library(udpipe)
#' ## Take data and standardise it a bit
#' data(brussels_reviews, package = "udpipe")
#' x <- subset(brussels_reviews, language == "nl")
#' x <- tolower(x$feedback)
#' 
#' ## Build the model get word embeddings and nearest neighbours
#' model <- word2vec(x = x, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' head(emb)
#' emb <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn  <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## Get vocabulary
#' vocab <- summary(model, type = "vocabulary")
#' 
#' # Do some calculations with the vectors and find similar terms to these
#' emb <- as.matrix(model)
#' vector <- emb["buurt", ] - emb["rustige", ] + emb["restaurants", ]
#' predict(model, vector, type = "nearest", top_n = 10)
#' 
#' vector <- emb["gastvrouw", ] - emb["gastvrij", ]
#' predict(model, vector, type = "nearest", top_n = 5)
#' 
#' vectors <- emb[c("gastheer", "gastvrouw"), ]
#' vectors <- rbind(vectors, avg = colMeans(vectors))
#' predict(model, vectors, type = "nearest", top_n = 10)
#' 
#' ## Save the model to hard disk
#' path <- "mymodel.bin"
#' \dontshow{
#' path <- tempfile(pattern = "w2v", fileext = ".bin")
#' }
#' write.word2vec(model, file = path)
#' model <- read.word2vec(path)
#' 
#' \dontshow{
#' file.remove(path)
#' }
#' 
#' 
#' ## 
#' ## Example getting word embeddings 
#' ##   which are different depending on the parts of speech tag
#' ## Look to the help of the udpipe R package 
#' ##   to get parts of speech tags on text
#' ## 
#' library(udpipe)
#' data(brussels_reviews_anno, package = "udpipe")
#' x <- subset(brussels_reviews_anno, language == "fr")
#' x <- subset(x, grepl(xpos, pattern = paste(LETTERS, collapse = "|")))
#' x$text <- sprintf("%s/%s", x$lemma, x$xpos)
#' x <- subset(x, !is.na(lemma))
#' x <- paste.data.frame(x, term = "text", group = "doc_id", collapse = " ")
#' x <- x$text
#' 
#' model <- word2vec(x = x, dim = 15, iter = 20, split = c(" ", ".\n?!"))
#' emb   <- as.matrix(model)
#' nn    <- predict(model, c("cuisine/NN", "rencontrer/VB"), type = "nearest")
#' nn
#' nn    <- predict(model, c("accueillir/VBN", "accueillir/VBG"), type = "nearest")
#' nn
word2vec <- function(x,
                     type = c("cbow", "skip-gram"),
                     dim = 50, window = 5L, 
                     iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                     split = c(" \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r", 
                               ".\n?!"),
                     stopwords = character(),
                     threads = 1L){
    type <- match.arg(type)
    stopw <- stopwords
    model <- file.path(tempdir(), "w2v.bin")
    if(length(stopw) == 0){
        stopw <- ""
    }
    file_stopwords <- tempfile()
    writeLines(stopw, file_stopwords)
    on.exit({
        if (file.exists(file_stopwords)) file.remove(file_stopwords)
    })
    if(length(x) == 1){
         file_train <- x
    }else{
        file_train <- tempfile(pattern = "textspace_", fileext = ".txt")
        on.exit({
            if (file.exists(file_stopwords)) file.remove(file_stopwords)
            if (file.exists(file_train)) file.remove(file_train)
        })
        writeLines(text = x, con = file_train)  
    }
    
    
    expTableSize <- 1000L
    expValueMax <- 6L
    min_count <- as.integer(min_count)
    dim <- as.integer(dim)
    window <- as.integer(window)
    iter <- as.integer(iter)
    expTableSize <- as.integer(expTableSize)
    expValueMax <- as.integer(expValueMax)
    sample <- as.numeric(sample)
    hs <- as.logical(hs)
    negative <- as.integer(negative)
    threads <- as.integer(threads)
    iter <- as.integer(iter)
    lr <- as.numeric(lr)
    skipgram <- as.logical(type %in% "skip-gram")
    split <- as.character(split)
    model <- w2v_train(trainFile = file_train, modelFile = model, stopWordsFile = file_stopwords,
                       minWordFreq = min_count,
                       size = dim, window = window, expTableSize = expTableSize, expValueMax = expValueMax, 
                       sample = sample, withHS = hs, negative = negative, threads = threads, iterations = iter,
                       alpha = lr, withSG = skipgram, wordDelimiterChars = split[1], endOfSentenceChars = split[2])
    model$data$stopwords <- stopwords
    model
}


#' @title Get the word vectors of a word2vec model
#' @description Get the word vectors of a word2vec model as a matrix.
#' @param x a word2vec model as returned by \code{\link{word2vec}} or \code{\link{read.word2vec}}
#' @param ... not used
#' @return a matrix with the word vectors where the rownames are the words from the model vocabulary
#' @export
#' @seealso \code{\link{word2vec}}, \code{\link{read.word2vec}}
#' @export
#' @examples 
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' 
#' embedding <- as.matrix(model)
as.matrix.word2vec <- function(x, ...){
    words <- w2v_dictionary(x$model)
    x <- w2v_embedding(x$model, words)
    x 
}


#' @export
as.matrix.word2vec_trained <- function(x, ...){
    as.matrix.word2vec(x)
}


#' @title Save a word2vec model to disk
#' @description Save a word2vec model as a binary file to disk or as a text file
#' @param x an object of class \code{w2v} or \code{w2v_trained} as returned by \code{\link{word2vec}}
#' @param file the path to the file where to store the model
#' @param type either 'bin' or 'txt' to write respectively the file as binary or as a text file. Defaults to 'bin'.
#' @return a logical indicating if the save process succeeded
#' @export
#' @seealso \code{\link{word2vec}}
#' @examples 
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' 
#' 
#' ## Save the model to hard disk as a binary file
#' path <- "mymodel.bin"
#' \dontshow{
#' path <- tempfile(pattern = "w2v", fileext = ".bin")
#' }
#' write.word2vec(model, file = path)
#' \dontshow{
#' file.remove(path)
#' }
#' ## Save the model to hard disk as a text file (uses package udpipe)
#' library(udpipe)
#' path <- "mymodel.txt"
#' \dontshow{
#' path <- tempfile(pattern = "w2v", fileext = ".txt")
#' }
#' write.word2vec(model, file = path, type = "txt")
#' \dontshow{
#' file.remove(path)
#' }
write.word2vec <- function(x, file, type = c("bin", "txt")){
    type <- match.arg(type)
    stopifnot(inherits(x, "w2v_trained") || inherits(x, "w2v"))
    if(type == "bin"){
        w2v_save_model(x$model, file)
    }else if(type == "txt"){
        requireNamespace(package = "udpipe")
        wordvectors <- as.matrix(x)
        wv <- udpipe::as_word2vec(wordvectors)
        f <- base::file(file, open = "wt", encoding = "UTF-8")
        cat(wv, file = f)
        close(f)
        file.exists(file)
    }
}

#' @title Read a binary word2vec model from disk
#' @description Read a binary word2vec model from disk
#' @param file the path to the model file
#' @return an object of class w2v which is a list with elements
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{model_path: the path to the model on disk}
#' \item{dim: the dimension of the embedding matrix}
#' \item{n: the number of words in the vocabulary}
#' }
#' @export
#' @examples
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' vocab <- summary(model, type = "vocabulary")
#' emb <- predict(model, c("bus", "naar", "unknownword"), type = "embedding")
#' emb
#' nn  <- predict(model, c("bus", "toilet"), type = "nearest")
#' nn
#' 
#' # Do some calculations with the vectors and find similar terms to these
#' emb <- as.matrix(model)
#' vector <- emb["gastvrouw", ] - emb["gastvrij", ]
#' predict(model, vector, type = "nearest", top_n = 5)
#' vectors <- emb[c("gastheer", "gastvrouw"), ]
#' vectors <- rbind(vectors, avg = colMeans(vectors))
#' predict(model, vectors, type = "nearest", top_n = 10)
read.word2vec <- function(file){
    stopifnot(file.exists(file))
    w2v_load_model(file)
}

#' @export
summary.word2vec <- function(object, type = "vocabulary", ...){
    type <- match.arg(type)
    if(type == "vocabulary"){
        w2v_dictionary(object$model)
    }else{
        stop("not implemented")
    }
}

#' @export
summary.word2vec_trained <- function(object, type = "vocabulary", ...){
    summary.word2vec(object = object, type = type, ...)
}



#' @title Predict functionalities for a word2vec model
#' @description Get either 
#' \itemize{
#' \item{the embedding of words}
#' \item{the nearest words which are similar to either a word or a word vector}
#' }
#' @param object a word2vec model as returned by \code{\link{word2vec}} or \code{\link{read.word2vec}}
#' @param newdata for type 'embedding', \code{newdata} should be a character vector of words\cr
#' for type 'nearest', \code{newdata} should be a character vector of words or a matrix in the embedding space
#' @param type either 'embedding' or 'nearest'. Defaults to 'nearest'.
#' @param top_n show only the top n nearest neighbours. Defaults to 10.
#' @param ... not used
#' @return depending on the type, you get a different result back:
#' \itemize{
#' \item{for type nearest: a list of data.frames with columns term, similarity and rank indicating with words which are closest to the provided \code{newdata} words or word vectors. If \code{newdata} is just one vector instead of a matrix, it returns a data.frame}
#' \item{for type embedding: a matrix of word vectors of the words provided in \code{newdata}}
#' }
#' @seealso \code{\link{word2vec}}, \code{\link{read.word2vec}}
#' @export
#' @examples 
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' emb <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn  <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' # Do some calculations with the vectors and find similar terms to these
#' emb <- as.matrix(model)
#' vector <- emb["buurt", ] - emb["rustige", ] + emb["restaurants", ]
#' predict(model, vector, type = "nearest", top_n = 10)
#' 
#' vector <- emb["gastvrouw", ] - emb["gastvrij", ]
#' predict(model, vector, type = "nearest", top_n = 5)
#' 
#' vectors <- emb[c("gastheer", "gastvrouw"), ]
#' vectors <- rbind(vectors, avg = colMeans(vectors))
#' predict(model, vectors, type = "nearest", top_n = 10)
predict.word2vec <- function(object, newdata, type = c("nearest", "embedding"), top_n = 10L, ...){
    type <- match.arg(type)
    top_n <- as.integer(top_n)
    if(type == "embedding"){
        x <- w2v_embedding(object$model, x = newdata)
    }else if(type == "nearest"){
        if(is.character(newdata)){
            x <- lapply(newdata, FUN=function(x, top_n, ...){
                w2v_nearest(object$model, x = x, top_n = top_n, ...)    
            }, top_n = top_n, ...)
            names(x) <- newdata    
        }else if(is.matrix(newdata)){
            x <- lapply(seq_len(nrow(newdata)), FUN=function(i, top_n, ...){
                w2v_nearest_vector(object$model, x = newdata[i, ], top_n = top_n, ...)    
            }, top_n = top_n, ...)
            if(!is.null(rownames(newdata))){
                names(x) <- rownames(newdata)    
            }
        }else if(is.numeric(newdata)){
            x <- w2v_nearest_vector(object$model, x = newdata, top_n = top_n, ...)    
        }
    }
    x
}

#' @export
predict.word2vec_trained <- function(object, newdata, type = c("nearest", "embedding"), ...){
    predict.word2vec(object = object, newdata = newdata, type = type, ...)
}



#' @title Similarity between word vectors as used in word2vec
#' @description The similarity between word vectors is defined as the square root of the average inner product of the vector elements (sqrt(sum(x . y) / ncol(x))) capped to zero
#' @param x a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param y a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param top_n integer indicating to return only the top n most similar terms from y for each row of x. 
#' If \code{top_n} is supplied, a data.frame will be returned with only the highest similarities between x and y 
#' instead of all pairwise similarities
#' @return 
#' By default, the function returns a similarity matrix between the rows of \code{x} and the rows of \code{y}. 
#' The similarity between row i of \code{x} and row j of \code{y} is found in cell \code{[i, j]} of the returned similarity matrix.\cr
#' If \code{top_n} is provided, the return value is a data.frame with columns term1, term2, similarity and rank 
#' indicating the similarity between the provided terms in \code{x} and \code{y} 
#' ordered from high to low similarity and keeping only the top_n most similar records.
#' @export
#' @seealso \code{\link{word2vec}}
#' @examples 
#' x <- matrix(rnorm(6), nrow = 2, ncol = 3)
#' rownames(x) <- c("word1", "word2")
#' y <- matrix(rnorm(15), nrow = 5, ncol = 3)
#' rownames(y) <- c("term1", "term2", "term3", "term4", "term5")
#' 
#' word2vec_similarity(x, y)
#' word2vec_similarity(x, y, top_n = 1)
#' word2vec_similarity(x, y, top_n = 2)
#' word2vec_similarity(x, y, top_n = +Inf)
#' 
#' ## Example with a word2vec model
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' emb <- as.matrix(model)
#' 
#' x <- emb[c("gastheer", "gastvrouw", "kamer"), ]
#' y <- emb
#' word2vec_similarity(x, x)
#' word2vec_similarity(x, y, top_n = 3)
#' predict(model, x, type = "nearest", top_n = 3)
word2vec_similarity <- function(x, y, top_n = +Inf){
    if (!is.matrix(x)) {
        x <- matrix(x, nrow = 1)
    }
    if (!is.matrix(y)) {
        y <- matrix(y, nrow = 1)
    }
    vectorsize <- ncol(x)
    similarities <- tcrossprod(x, y)
    similarities <- similarities / vectorsize
    similarities[similarities < 0] <- 0
    similarities <- sqrt(similarities)
    if (!missing(top_n)) {
        similarities <- as.data.frame.table(similarities, stringsAsFactors = FALSE)
        colnames(similarities) <- c("term1", "term2", "similarity")
        similarities <- similarities[order(factor(similarities$term1), similarities$similarity, decreasing = TRUE), ]
        similarities$rank <- stats::ave(similarities$similarity, similarities$term1, FUN = seq_along)
        similarities <- similarities[similarities$rank <= top_n, ]
        rownames(similarities) <- NULL
    }
    similarities
}