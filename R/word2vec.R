#' @title Train a word2vec model on text
#' @description Construct a word2vec model on text. The algorithm is explained at \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @param x a character vector with text or the path to the file on disk containing training data or a list of tokens. See the examples.
#' @param type the type of algorithm to use, either 'cbow' or 'skip-gram'. Defaults to 'cbow'
#' @param dim dimension of the word vectors. Defaults to 50.
#' @param iter number of training iterations. Defaults to 5.
#' @param lr initial learning rate also known as alpha. Defaults to 0.05
#' @param window skip length between words. Defaults to 5.
#' @param hs logical indicating to use hierarchical softmax instead of negative sampling. Defaults to FALSE indicating to do negative sampling.
#' @param negative integer with the number of negative samples. Only used in case hs is set to FALSE
#' @param sample threshold for occurrence of words. Defaults to 0.001
#' @param min_count integer indicating the number of time a word should occur to be considered as part of the training vocabulary. Defaults to 5.
#' @param stopwords a character vector of stopwords to exclude from training 
#' @param threads number of CPU threads to use. Defaults to 1.
#' @param ... further arguments passed on to the methods \code{\link{word2vec.character}}, \code{\link{word2vec.list}} as well as the C++ function \code{w2v_train} - for expert use only
#' @return an object of class \code{w2v_trained} which is a list with elements 
#' \itemize{
#' \item{model: a Rcpp pointer to the model}
#' \item{data: a list with elements file: the training data used, stopwords: the character vector of stopwords, n}
#' \item{vocabulary: the number of words in the vocabulary}
#' \item{success: logical indicating if training succeeded}
#' \item{error_log: the error log in case training failed}
#' \item{control: as list of the training arguments used, namely min_count, dim, window, iter, lr, skipgram, hs, negative, sample, split_words, split_sents, expTableSize and expValueMax}
#' }
#' @references \url{https://github.com/maxoodf/word2vec}, \url{https://arxiv.org/pdf/1310.4546.pdf}
#' @details 
#' Some advice on the optimal set of parameters to use for training as defined by Mikolov et al.
#' \itemize{
#' \item{argument type: skip-gram (slower, better for infrequent words) vs cbow (fast)}
#' \item{argument hs: the training algorithm: hierarchical softmax (better for infrequent words) vs negative sampling (better for frequent words, better with low dimensional vectors)}
#' \item{argument dim: dimensionality of the word vectors: usually more is better, but not always}
#' \item{argument window: for skip-gram usually around 10, for cbow around 5}
#' \item{argument sample: sub-sampling of frequent words: can improve both accuracy and speed for large data sets (useful values are in range 0.001 to 0.00001)}
#' }
#' @note
#' Some notes on the tokenisation
#' \itemize{
#' \item{If you provide to \code{x} a list, each list element should correspond to a sentence (or what you consider as a sentence) and should contain a character vector of tokens. The word2vec model is then executed using \code{\link{word2vec.list}}}
#' \item{If you provide to \code{x} a character vector or the path to the file on disk, the tokenisation into words depends on the first element provided in \code{split} and the tokenisation into sentences depends on the second element provided in \code{split} when passed on to \code{\link{word2vec.character}}}
#' }
#' @seealso \code{\link{predict.word2vec}}, \code{\link{as.matrix.word2vec}}, \code{\link{word2vec}}, \code{\link{word2vec.character}}, \code{\link{word2vec.list}}
#' @export
#' @examples
#' \dontshow{if(require(udpipe))\{}
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
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## Get vocabulary
#' vocab   <- summary(model, type = "vocabulary")
#' 
#' # Do some calculations with the vectors and find similar terms to these
#' emb     <- as.matrix(model)
#' vector  <- emb["buurt", ] - emb["rustige", ] + emb["restaurants", ]
#' predict(model, vector, type = "nearest", top_n = 10)
#' 
#' vector  <- emb["gastvrouw", ] - emb["gastvrij", ]
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
#' ## 
#' ## Example of word2vec with a list of tokens 
#' ## 
#' toks  <- strsplit(x, split = "[[:space:][:punct:]]+")
#' model <- word2vec(x = toks, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
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
#' x <- split(x$text, list(x$doc_id, x$sentence_id))
#' 
#' model <- word2vec(x = x, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' nn    <- predict(model, c("cuisine/NN", "rencontrer/VB"), type = "nearest")
#' nn
#' nn    <- predict(model, c("accueillir/VBN", "accueillir/VBG"), type = "nearest")
#' nn
#' 
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
word2vec <- function(x, 
                     type = c("cbow", "skip-gram"),
                     dim = 50, window = ifelse(type == "cbow", 5L, 10L), 
                     iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                     stopwords = character(),
                     threads = 1L,
                     ...) {
    UseMethod("word2vec")
}

#' @inherit word2vec title description params details seealso return references
#' @export
#' @examples 
#' \dontshow{if(require(udpipe))\{}
#' library(udpipe)
#' data(brussels_reviews, package = "udpipe")
#' x     <- subset(brussels_reviews, language == "nl")
#' x     <- tolower(x$feedback)
#' toks  <- strsplit(x, split = "[[:space:][:punct:]]+")
#' model <- word2vec(x = toks, dim = 15, iter = 20)
#' emb   <- as.matrix(model)
#' head(emb)
#' emb   <- predict(model, c("bus", "toilet", "unknownword"), type = "embedding")
#' emb
#' nn    <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
#' nn
#' 
#' ## 
#' ## Example of word2vec with a list of tokens
#' ## which gives the same embeddings as with a similarly tokenised character vector of texts 
#' ## 
#' txt   <- txt_clean_word2vec(x, ascii = TRUE, alpha = TRUE, tolower = TRUE, trim = TRUE)
#' table(unlist(strsplit(txt, "")))
#' toks  <- strsplit(txt, split = " ")
#' set.seed(1234)
#' modela <- word2vec(x = toks, dim = 15, iter = 20)
#' set.seed(1234)
#' modelb <- word2vec(x = txt, dim = 15, iter = 20, split = c(" \n\r", "\n\r"))
#' all.equal(as.matrix(modela), as.matrix(modelb))
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
word2vec.list <- function(x,
                          type = c("cbow", "skip-gram"),
                          dim = 50, window = ifelse(type == "cbow", 5L, 10L), 
                          iter = 5L, lr = 0.05, hs = FALSE, negative = 5L, sample = 0.001, min_count = 5L, 
                          stopwords = integer(),
                          threads = 1L,
                          ...){
    #x <- lapply(x, as.character)
    type <- match.arg(type)
    stopwords <- as.integer(stopwords)
    model <- file.path(tempdir(), "w2v.bin")
    #expTableSize <- 1000L
    #expValueMax <- 6L
    #expTableSize <- as.integer(expTableSize)
    #expValueMax <- as.integer(expValueMax)
    min_count <- as.integer(min_count)
    dim <- as.integer(dim)
    window <- as.integer(window)
    iter <- as.integer(iter)
    sample <- as.numeric(sample)
    hs <- as.logical(hs)
    negative <- as.integer(negative)
    threads <- as.integer(threads)
    iter <- as.integer(iter)
    lr <- as.numeric(lr)
    skipgram <- as.logical(type %in% "skip-gram")
   
    vocaburary <- unique(unlist(x, use.names = FALSE))
    vocaburary <- setdiff(vocaburary, stopwords)
    x <- lapply(x, function(x) {
        v <- fastmatch::fmatch(x, vocaburary)
        v[is.na(v)] <- 0L
        return(v)
    })
    model <- w2v_train(x, vocaburary, minWordFreq = min_count,
                       size = dim, window = window, #expTableSize = expTableSize, expValueMax = expValueMax, 
                       sample = sample, withHS = hs, negative = negative, threads = threads, iterations = iter,
                       alpha = lr, withSG = skipgram, ...)
    model$data$stopwords <- stopwords
    model
}


#' @title Get the word vectors of a word2vec model
#' @description Get the word vectors of a word2vec model as a dense matrix.
#' @param x a word2vec model as returned by \code{\link{word2vec}} or \code{\link{read.word2vec}}
#' @param encoding set the encoding of the row names to the specified encoding. Defaults to 'UTF-8'.
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
as.matrix.word2vec <- function(x, encoding='UTF-8', ...){
    words <- w2v_dictionary(x$model)
    x <- w2v_embedding(x$model, words)
    Encoding(rownames(x)) <- encoding
    x 
}


#' @export
as.matrix.word2vec_trained <- function(x, encoding='UTF-8', ...){
    as.matrix.word2vec(x)
}


#' @title Save a word2vec model to disk
#' @description Save a word2vec model as a binary file to disk or as a text file
#' @param x an object of class \code{w2v} or \code{w2v_trained} as returned by \code{\link{word2vec}}
#' @param file the path to the file where to store the model
#' @param type either 'bin' or 'txt' to write respectively the file as binary or as a text file. Defaults to 'bin'.
#' @param encoding encoding to use when writing a file with type 'txt' to disk. Defaults to 'UTF-8'
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
#' 
#' \dontshow{if(require(udpipe))\{}
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
#' \dontshow{\} # End of main if statement running only if the required packages are installed}
write.word2vec <- function(x, file, type = c("bin", "txt"), encoding = "UTF-8"){
    type <- match.arg(type)
    if(type == "bin"){
        stopifnot(inherits(x, "w2v_trained") || inherits(x, "w2v") || inherits(x, "word2vec_trained") || inherits(x, "word2vec"))
        w2v_save_model(x$model, file)
    }else if(type == "txt"){
        requireNamespace(package = "udpipe")
        wordvectors <- as.matrix(x)
        wv <- udpipe::as_word2vec(wordvectors)
        f <- base::file(file, open = "wt", encoding = encoding)
        cat(wv, file = f)
        close(f)
        file.exists(file)
    }
}

#' @title Read a binary word2vec model from disk
#' @description Read a binary word2vec model from disk
#' @param file the path to the model file
#' @param normalize logical indicating to normalize the embeddings by dividing by the factor (sqrt(sum(x . x) / length(x))). Defaults to FALSE. 
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
read.word2vec <- function(file, normalize = FALSE){
    stopifnot(file.exists(file))
    w2v_load_model(file, normalize = normalize)    
}


#' @title Read word vectors from a word2vec model from disk
#' @description Read word vectors from a word2vec model from disk into a dense matrix
#' @param file the path to the model file
#' @param type either 'bin' or 'txt' indicating the \code{file} is a binary file or a text file
#' @param n integer, indicating to limit the number of words to read in. Defaults to reading all words.
#' @param normalize logical indicating to normalize the embeddings by dividing by the factor (sqrt(sum(x . x) / length(x))). Defaults to FALSE. 
#' @param encoding encoding to be assumed for the words. Defaults to 'UTF-8'
#' @return A matrix with the embeddings of the words. The rownames of the matrix are the words which are by default set to UTF-8 encoding.
#' @export
#' @examples
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' embed <- read.wordvectors(path, type = "bin", n = 10)
#' embed <- read.wordvectors(path, type = "bin", n = 10, normalize = TRUE)
#' embed <- read.wordvectors(path, type = "bin")
#' 
#' path  <- system.file(package = "word2vec", "models", "example.txt")
#' embed <- read.wordvectors(path, type = "txt", n = 10)
#' embed <- read.wordvectors(path, type = "txt", n = 10, normalize = TRUE)
#' embed <- read.wordvectors(path, type = "txt")
read.wordvectors <- function(file, type = c("bin", "txt"), n = .Machine$integer.max, normalize = FALSE, encoding = "UTF-8"){
    type <- match.arg(type)
    if(type == "bin"){
        x <- w2v_read_binary(file, normalize = normalize, n = as.integer(n))
        Encoding(rownames(x)) <- encoding
        x    
    }else if(type == "txt"){
        if(n < .Machine$integer.max){
            x <- readLines(file, skipNul = TRUE, encoding = encoding, n = n + 1L, warn = FALSE)
        }else{
            x <- readLines(file, skipNul = TRUE, encoding = encoding, warn = FALSE)
        }
        size <- x[1]
        size <- as.numeric(unlist(strsplit(size, " ")))
        x <- x[-1]
        x <- strsplit(x, " ")
        size[1] <- length(x)
        token <- sapply(x, FUN=function(x) x[1])
        emb <- lapply(x, FUN=function(x) as.numeric(x[-1]))
        embedding <- matrix(data = unlist(emb), nrow = size[1], ncol = size[2], dimnames = list(token), byrow = TRUE)
        if(normalize){
            embedding <- t(apply(embedding, MARGIN=1, FUN=function(x) x / sqrt(sum(x * x) / length(x))))
        }
        embedding
    }
}

#' @export
summary.word2vec <- function(object, type = "vocabulary", encoding = "UTF-8", ...){
    type <- match.arg(type)
    if(type == "vocabulary"){
        x <- w2v_dictionary(object$model)
        Encoding(x) <- encoding
        x
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
#' @param encoding set the encoding of the text elements to the specified encoding. Defaults to 'UTF-8'. 
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
predict.word2vec <- function(object, newdata, type = c("nearest", "embedding"), top_n = 10L, encoding = "UTF-8", ...){
    type <- match.arg(type)
    top_n <- as.integer(top_n)
    if(type == "embedding"){
        x <- w2v_embedding(object$model, x = newdata)
        Encoding(rownames(x)) <- encoding
    }else if(type == "nearest"){
        if(is.character(newdata)){
            x <- lapply(newdata, FUN=function(x, top_n, ...){
                data <- w2v_nearest(object$model, x = x, top_n = top_n, ...)    
                Encoding(data$term1) <- encoding
                Encoding(data$term2) <- encoding
                data
            }, top_n = top_n, ...)
            names(x) <- newdata    
        }else if(is.matrix(newdata)){
            x <- lapply(seq_len(nrow(newdata)), FUN=function(i, top_n, ...){
                data <- w2v_nearest_vector(object$model, x = newdata[i, ], top_n = top_n, ...)    
                Encoding(data$term) <- encoding
                data
            }, top_n = top_n, ...)
            if(!is.null(rownames(newdata))){
                names(x) <- rownames(newdata)    
            }
        }else if(is.numeric(newdata)){
            x <- w2v_nearest_vector(object$model, x = newdata, top_n = top_n, ...)    
            Encoding(x$term) <- encoding
        }
    }
    x
}

#' @export
predict.word2vec_trained <- function(object, newdata, type = c("nearest", "embedding"), ...){
    predict.word2vec(object = object, newdata = newdata, type = type, ...)
}



#' @title Similarity between word vectors as used in word2vec
#' @description The similarity between word vectors is defined 
#' \itemize{
#'  \item{for type 'dot': as the square root of the average inner product of the vector elements (sqrt(sum(x . y) / ncol(x))) capped to zero}
#'  \item{for type 'cosine': as the the cosine similarity, namely sum(x . y) / (sum(x^2)*sum(y^2)) }
#' }
#' @param x a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param y a matrix with embeddings where the rownames of the matrix provide the label of the term
#' @param top_n integer indicating to return only the top n most similar terms from y for each row of x. 
#' If \code{top_n} is supplied, a data.frame will be returned with only the highest similarities between x and y 
#' instead of all pairwise similarities
#' @param type character string with the type of similarity. Either 'dot' or 'cosine'. Defaults to 'dot'.
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
#' word2vec_similarity(x, y, type = "cosine")
#' word2vec_similarity(x, y, top_n = 1, type = "cosine")
#' word2vec_similarity(x, y, top_n = 2, type = "cosine")
#' word2vec_similarity(x, y, top_n = +Inf, type = "cosine")
#' 
#' ## Example with a word2vec model
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' emb   <- as.matrix(model)
#' 
#' x <- emb[c("gastheer", "gastvrouw", "kamer"), ]
#' y <- emb
#' word2vec_similarity(x, x)
#' word2vec_similarity(x, y, top_n = 3)
#' predict(model, x, type = "nearest", top_n = 3)
word2vec_similarity <- function(x, y, top_n = +Inf, type = c("dot", "cosine")){
    type <- match.arg(type)
    if (!is.matrix(x)) {
        x <- matrix(x, nrow = 1)
    }
    if (!is.matrix(y)) {
        y <- matrix(y, nrow = 1)
    }
    if(type == "dot"){
        vectorsize   <- ncol(x)
        similarities <- tcrossprod(x, y)
        similarities <- similarities / vectorsize
        similarities[similarities < 0] <- 0
        similarities <- sqrt(similarities)    
    }else if (type == "cosine"){
        similarities  <- tcrossprod(x, y)
        x_scale       <- sqrt(apply(x, MARGIN = 1, FUN = crossprod))
        y_scale       <- sqrt(apply(y, MARGIN = 1, FUN = crossprod))
        similarities  <- similarities / outer(x_scale, y_scale, FUN = "*")
    }
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

