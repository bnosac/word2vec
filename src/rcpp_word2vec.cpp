#include <Rcpp.h>
#include <progress.hpp>
#include <progress_bar.hpp>

#include <iostream>
#include <iomanip>
#include "word2vec.hpp"
#include <unordered_map>

// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::export]]
Rcpp::List w2v_train(std::string trainFile, std::string modelFile, std::string stopWordsFile, 
                     uint16_t minWordFreq = 5,
                     uint16_t size = 100,
                     uint8_t window = 5,
                     uint16_t expTableSize = 1000,
                     uint8_t expValueMax = 6,
                     float sample = 0.001,
                     bool withHS = false,
                     uint8_t negative = 5,
                     uint8_t threads = 1,
                     uint8_t iterations = 5,
                     float alpha = 0.05,
                     bool withSG = false,
                     std::string wordDelimiterChars = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r",
                     std::string endOfSentenceChars = ".\n?!",
                     bool verbose = false) {
  
  
  /*
   uint16_t minWordFreq = 5; ///< discard words that appear less than minWordFreq times
   uint16_t size = 100; ///< word vector size
   uint8_t window = 5; ///< skip length between words
   uint16_t expTableSize = 1000; ///< exp(x) / (exp(x) + 1) values lookup table size
   uint8_t expValueMax = 6; ///< max value in the lookup table
   float sample = 1e-3f; ///< threshold for occurrence of words
   bool withHS = false; ///< use hierarchical softmax instead of negative sampling
   uint8_t negative = 5; ///< negative examples number
   uint8_t threads = 12; ///< train threads number
   uint8_t iterations = 5; ///< train iterations
   float alpha = 0.05f; ///< starting learn rate
   bool withSG = false; ///< use Skip-Gram instead of CBOW
   std::string wordDelimiterChars = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r";
   std::string endOfSentenceChars = ".\n?!";
   */
  w2v::trainSettings_t trainSettings;
  trainSettings.minWordFreq = minWordFreq;
  trainSettings.size = size;
  trainSettings.window = window;
  trainSettings.expTableSize = expTableSize;
  trainSettings.expValueMax = expValueMax;
  trainSettings.sample = sample;
  trainSettings.withHS = withHS;
  trainSettings.negative = negative;
  trainSettings.threads = threads;
  trainSettings.iterations = iterations;
  trainSettings.alpha = alpha;
  trainSettings.withSG = withSG;
  trainSettings.wordDelimiterChars = wordDelimiterChars;
  trainSettings.endOfSentenceChars = endOfSentenceChars;
  Rcpp::XPtr<w2v::w2vModel_t> model(new w2v::w2vModel_t(), true);
  bool trained;
  
  std::size_t vocWords;
  std::size_t trainWords;
  std::size_t totalWords;
  if (verbose) {
    Progress p(100, true);
    trained = model->train(trainSettings, trainFile, stopWordsFile,
                           [&p] (float _percent) {
                             p.update(_percent/2);
                             /*
                              std::cout << "\rParsing train data... "
                                        << std::fixed << std::setprecision(2)
                                        << _percent << "%" << std::flush;
                              */
                           },
                           [&vocWords, &trainWords, &totalWords] (std::size_t _vocWords, std::size_t _trainWords, std::size_t _totalWords) {
                             /*
                              Rcpp::Rcerr << std::endl
                                          << "Finished reading data: " << std::endl
                                          << "Vocabulary size: " << _vocWords << std::endl
                                          << "Train words: " << _trainWords << std::endl
                                          << "Total words: " << _totalWords << std::endl
                                          << "Start training" << std::endl
                                          << std::endl;
                              */
                             vocWords = _vocWords;
                             trainWords = _trainWords;
                             totalWords = _totalWords;
                           },
                           [&p] (float _alpha, float _percent) {
                             /*
                              std::cout << '\r'
                                        << "alpha: "
                                        << std::fixed << std::setprecision(6)
                                        << _alpha
                                        << ", progress: "
                                        << std::fixed << std::setprecision(2)
                                        << _percent << "%"
                                        << std::flush;
                              */
                             p.update(50+(_percent/2));
                           }
    );
    //std::cout << std::endl;
  } else {
    trained = model->train(trainSettings, trainFile, stopWordsFile, 
                           nullptr, 
                           [&vocWords, &trainWords, &totalWords] (std::size_t _vocWords, std::size_t _trainWords, std::size_t _totalWords) {
                             /*
                              Rcpp::Rcerr << std::endl
                                          << "Finished reading data: " << std::endl
                                          << "Vocabulary size: " << _vocWords << std::endl
                                          << "Train words: " << _trainWords << std::endl
                                          << "Total words: " << _totalWords << std::endl
                                          << "Start training" << std::endl
                                          << std::endl;
                              */
                             vocWords = _vocWords;
                             trainWords = _trainWords;
                             totalWords = _totalWords;
                           },
                           nullptr);
  }
  bool success = true;
  if (!trained) {
    Rcpp::Rcout << "Training failed: " << model->errMsg() << std::endl;
    success = false;
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("data") = Rcpp::List::create(
      Rcpp::Named("file") = trainFile,
      Rcpp::Named("stopwords") = stopWordsFile,
      Rcpp::Named("n") = totalWords,
      Rcpp::Named("n_vocabulary") = trainWords
    ),
    Rcpp::Named("vocabulary") = vocWords,
    Rcpp::Named("success") = success,
    Rcpp::Named("error_log") = model->errMsg(),
    Rcpp::Named("control") = Rcpp::List::create(
      Rcpp::Named("min_count") = minWordFreq,
      Rcpp::Named("dim") = size,
      Rcpp::Named("window") = window,
      Rcpp::Named("iter") = iterations,
      Rcpp::Named("lr") = alpha,
      Rcpp::Named("skipgram") = withSG,
      Rcpp::Named("hs") = withHS,
      Rcpp::Named("negative") = negative,
      Rcpp::Named("sample") = sample,
      Rcpp::Named("expTableSize") = expTableSize,
      Rcpp::Named("expValueMax") = expValueMax,
      Rcpp::Named("split_words") = wordDelimiterChars,
      Rcpp::Named("split_sents") = endOfSentenceChars
    )
  );
  out.attr("class") = "w2v_trained";
  return out;
}


// [[Rcpp::export]]
Rcpp::List w2v_load_model(std::string file) {
  Rcpp::XPtr<w2v::w2vModel_t> model(new w2v::w2vModel_t(), true);
  if (!model->load(file)) {
    Rcpp::stop(model->errMsg());
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("model_path") = file,
    Rcpp::Named("dim") = model->vectorSize(),
    Rcpp::Named("vocabulary") = model->modelSize()
  );
  out.attr("class") = "w2v";
  return out;
}

// [[Rcpp::export]]
bool w2v_save_model(SEXP ptr, std::string file) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  bool success = model->save(file);
  return success;
}

// [[Rcpp::export]]
std::vector<std::string> w2v_dictionary(SEXP ptr) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  
  std::unordered_map<std::string, w2v::vector_t> m_map = model->map();
  std::vector<std::string> keys;
  for(auto kv : m_map) {
    keys.push_back(kv.first);
  } 
  return keys;
}


// [[Rcpp::export]]
Rcpp::NumericMatrix w2v_embedding(SEXP ptr, Rcpp::StringVector x) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  Rcpp::NumericMatrix embedding(x.size(), model->vectorSize());
  rownames(embedding) = x;
  std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
  
  for (int i = 0; i < x.size(); i++){
    std::string input = Rcpp::as<std::string>(x[i]);
    auto vec = model->vector(input);
    if (vec != nullptr) {
      for(unsigned int j = 0; j < vec->size(); j++){
        embedding(i, j) = (float)((*vec)[j]);
      }
    }
  }
  return embedding;
}

// [[Rcpp::export]]
Rcpp::DataFrame w2v_nearest(SEXP ptr, 
                            std::string x, 
                            std::size_t top_n = 10,
                            float min_distance = 0.0) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  std::unordered_map<std::string, w2v::vector_t> m_map = model->map();
  auto const &i = m_map.find(x);
  w2v::vector_t vec;
  if (i != m_map.end()) {
    vec = i->second;
  }else{
    Rcpp::stop("Could not find the word in the dictionary: " + x);
  }
  std::vector<std::pair<std::string, float>> nearest;
  model->nearest(vec, nearest, top_n, min_distance);
  
  std::vector<std::string> keys;
  std::vector<float> distance;
  std::vector<int> rank;
  int r = 0;
  for(auto kv : nearest) {
    keys.push_back(kv.first);
    distance.push_back(kv.second);
    r = r + 1;
    rank.push_back(r);
  } 
  Rcpp::DataFrame out = Rcpp::DataFrame::create(
    Rcpp::Named("term1") = x,
    Rcpp::Named("term2") = keys,
    Rcpp::Named("similarity") = distance,
    Rcpp::Named("rank") = rank
  );
  return out;
}




// [[Rcpp::export]]
Rcpp::List w2v_analogy(SEXP ptr, 
                       Rcpp::StringVector x, 
                       std::size_t n = 10,
                       float min_distance = 0.0) {
  Rcpp::List out;
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  std::vector<std::pair<std::string, float>> nearest;
  
  std::string word1, word2, word3;
  
  try {
  word1 = Rcpp::as<std::string>(x[0]);
  word2 = Rcpp::as<std::string>(x[1]);
  word3 = Rcpp::as<std::string>(x[2]);
  //w2v::word2vec_t vec1(model, &word1);
  //w2v::word2vec_t vec2(model, &word2);
  //w2v::word2vec_t vec3(model, &word3);

  w2v::vector_t vec1 = *(model->vector(word1));
  w2v::vector_t vec2 = *(model->vector(word2));
  w2v::vector_t vec3 = *(model->vector(word3));
  w2v::vector_t vec = vec2 - vec1 + vec3;
  //w2v::vector_t result = king - man + woman;
  
  model->nearest(vec, nearest, n, min_distance);
  
  std::vector<std::string> keys;
  std::vector<float> distance;
  for(auto kv : nearest) {
    keys.push_back(kv.first);
    distance.push_back(kv.second);
  } 
  out = Rcpp::List::create(
    Rcpp::Named("term") = keys,
    Rcpp::Named("distance") = distance
  );
  } catch (const std::exception &_e) {
    Rcpp::stop(_e.what());
  } catch (...) {
    Rcpp::stop("unknown error");
  }
  return out;
}


  
  
  
