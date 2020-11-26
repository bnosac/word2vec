#' @title Text cleaning specific for input to word2vec
#' @description Standardise text by
#' \itemize{
#' \item{Conversion of text from UTF-8 to ASCII}
#' \item{Keeping only alphanumeric characters: letters and numbers}
#' \item{Removing multiple spaces}
#' \item{Removing leading/trailing spaces}
#' \item{Performing lowercasing}
#' }
#' @param x a character vector in UTF-8 encoding
#' @param ascii logical indicating to use \code{iconv} to convert the input from UTF-8 to ASCII. Defaults to TRUE.
#' @param alpha logical indicating to keep only alphanumeric characters. Defaults to TRUE.
#' @param tolower logical indicating to lowercase \code{x}. Defaults to TRUE.
#' @param trim logical indicating to trim leading/trailing white space. Defaults to TRUE.
#' @return a character vector of the same length as \code{x} 
#' which is standardised by converting the encoding to ascii, lowercasing and 
#' keeping only alphanumeric elements 
#' @export
#' @examples 
#' x <- c("  Just some.texts,  ok?", "123.456 and\tsome MORE!  ")
#' txt_clean_word2vec(x)
txt_clean_word2vec <- function(x, ascii = TRUE, alpha = TRUE, tolower = TRUE, trim = TRUE){
    text <- x
    if(ascii){
        text <- iconv(text, from = "UTF-8", to = "ASCII//TRANSLIT")   
    }
    if(alpha){
        text <- gsub("[^[:alnum:]]", " ", text)    
    }
    text <- gsub(" +", " ", text)
    if(tolower){
        text <- tolower(text)    
    }
    if(trim){
        text <- trimws(text)   
    }
    text
}