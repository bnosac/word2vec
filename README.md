# word2vec 

This repository contains an R package allowing to build a word2vec model

- It is based on the paper *Distributed Representations of Words and Phrases and their Compositionality* [[Mikolov et al.](https://arxiv.org/pdf/1310.4546.pdf)]
- This R package is an Rcpp wrapper around https://github.com/maxoodf/word2vec
- The package allows one 
    - to train word embeddings using multiple threads on character data or data in a text file
    - use the embeddings to find relations between words

## Installation

- For regular users, install the package from your local CRAN mirror `install.packages("word2vec")`
- For installing the development version of this package: `remotes::install_github("bnosac/word2vec")`

Look to the documentation of the functions

```{r}
help(package = "word2vec")
```

## Example

- Take some data and standardise it a bit

```{r}
library(udpipe)
data(brussels_reviews, package = "udpipe")
x <- subset(brussels_reviews, language == "nl")
x <- tolower(x$feedback)
```

- Build a model

```{r}
library(word2vec)
model <- word2vec(x = x, type = "cbow", dim = 15, iter = 20)
embedding <- as.matrix(model)
embedding <- predict(model, c("bus", "toilet"), type = "embedding")
lookslike <- predict(model, c("bus", "toilet"), type = "nearest", top_n = 5)
lookslike
$bus
 term1       term2 similarity rank
   bus       lopen  0.7737458    1
   bus     minuten  0.7378477    2
   bus     centrum  0.7323325    3
   bus loopafstand  0.7200720    4
   bus       markt  0.7054080    5

$toilet
  term1     term2 similarity rank
 toilet  voorzien  0.6782151    1
 toilet  gemakken  0.6778656    2
 toilet ingericht  0.6666461    3
 toilet    netjes  0.6460885    4
 toilet  badkamer  0.6415362    5
```

- Save the model and read it back in and do something with it

```{r}
write.word2vec(model, "mymodel.bin")
model     <- read.word2vec("mymodel.bin")
terms     <- summary(model, "vocabulary")
embedding <- as.matrix(model)
```

## Support in text mining

Need support in text mining?
Contact BNOSAC: http://www.bnosac.be

