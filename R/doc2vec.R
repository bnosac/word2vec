

embed_doc <- function(model, tokens, encoding = "UTF-8"){
    ## Get embedding of the tokens
    emb <- predict(model, tokens, "embedding", encoding = encoding)
    emb <- emb[which(!is.na(emb[, 1])), , drop = FALSE]
    if(nrow(emb) == 0){
        emb <- rep(NA_real_, ncol(emb))
        return(emb)
    }
    ## Sum the embeddings and standardise 
    emb <- colSums(emb, na.rm = TRUE)
    emb <- emb / sqrt(sum(emb^2) / length(emb))
    emb
}


#' @title Get document vectors based on a word2vec model
#' @description Document vectors are the sum of the vectors of the words which are part of the document standardised by the scale of the vector space.
#' This scale is the sqrt of the average inner product of the vector elements.
#' @param object a word2vec model as returned by \code{\link{word2vec}} or \code{\link{read.word2vec}}
#' @param newdata either a  list of tokens where each list element is a character vector of tokens which form the document and the list name is considered the document identifier; 
#' or a data.frame with columns doc_id and text; or a character vector with texts where the character vector names will be considered the document identifier
#' @param split in case \code{newdata} is not a list of tokens, text will be splitted into tokens by splitting based on function \code{\link{strsplit}} with the provided \code{split} argument
#' @param encoding set the encoding of the text elements to the specified encoding. Defaults to 'UTF-8'. 
#' @param ... not used
#' @return a matrix with 1 row per document containing the text document vectors, the rownames of this matrix are the document identifiers
#' @seealso \code{\link{word2vec}}, \code{\link{predict.word2vec}}
#' @export
#' @examples 
#' path  <- system.file(package = "word2vec", "models", "example.bin")
#' model <- read.word2vec(path)
#' x <- data.frame(doc_id = c("doc1", "doc2", "testmissingdata"), 
#'                 text = c("there is no toilet. on the bus", "no tokens from dictionary", NA),
#'                 stringsAsFactors = FALSE)
#' emb <- doc2vec(model, x, type = "embedding")
#' emb
#' word2vec_similarity(emb, doc2vec("i like busses with a toilet"))
#' 
#' ## similar way of extracting embeddings
#' x <- setNames(c("there is no toilet. on the bus", "no tokens from dictionary", NA), c("a", "b", "c"))
#' emb <- doc2vec(model, x, type = "embedding")
#' emb
#' 
#' ## similar way of extracting embeddings
#' x <- setNames(c("there is no toilet. on the bus", "no tokens from dictionary", NA), c("a", "b", "c"))
#' x <- strsplit(x, "[ .]")
#' emb <- doc2vec(model, x, type = "embedding")
#' emb
#' 
#' ## show behaviour in case of NA or character data of no length
#' x <- list(a = character(), b = c("bus", "toilet"), c = NA)
#' emb <- doc2vec(model, x, type = "embedding")
#' emb
doc2vec <- function(object, newdata, split = " ", encoding = "UTF-8", ...){
    if(!inherits(object, "word2vec")){
        warning("doc2vec requires as input an object of class word2vec")
    }
    if(is.character(newdata)){
        newdata <- strsplit(newdata, split = split)
    }else if(is.data.frame(newdata) && all(c("doc_id", "text") %in% colnames(newdata))){
        txt <- as.character(newdata$text)
        names(txt) <- newdata$doc_id
        newdata <- strsplit(txt, split)
    }else{
        stopifnot(is.list(newdata))
    }
    embedding <- lapply(newdata, FUN=function(x){
        if(length(x) == 0){
            return(rep(NA_real_, object$dim))
        }
        embed_doc(object, x, encoding = encoding)
    })
    embedding <- do.call(rbind, embedding)
    embedding
}