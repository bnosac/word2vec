#include <Rcpp.h>
#include <progress.hpp>
#include <progress_bar.hpp>

#include <iostream>
#include <iomanip>
#include "word2vec.hpp"
#include <unordered_map>

// [[Rcpp::depends(RcppProgress)]]
// [[Rcpp::export]]
Rcpp::List w2v_train(Rcpp::List texts_, 
                     Rcpp::CharacterVector types_, 
                     std::string modelFile = "", 
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
                     bool verbose = false,
                     bool normalize = true) {
  
  
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
  
  texts_t texts = Rcpp::as<texts_t>(texts_);
  types_t types = Rcpp::as<types_t>(types_);
  
  w2v::corpus_t corpus(texts, types);
  corpus.setWordFreq();
      
  // Rcpp::List out2 = Rcpp::List::create(
  //     Rcpp::Named("frequency") = corpus.frequency
  // );
  // 
  // return out2;
  
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
  //trainSettings.wordDelimiterChars = wordDelimiterChars;
  //trainSettings.endOfSentenceChars = endOfSentenceChars;
  Rcpp::XPtr<w2v::w2vModel_t> model(new w2v::w2vModel_t(), true);
  bool trained;
  
  std::size_t vocWords;
  std::size_t trainWords;
  std::size_t totalWords;
  if (verbose) { // NOTE: consider removing progress bar
    Progress p(100, true);
    trained = model->train(trainSettings, corpus, 
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
                             p.update(_percent);
                           }
    );
  } else {
    trained = model->train(trainSettings, corpus, nullptr);
  }
  Rcpp::Rcout << "Training done\n";
  //return Rcpp::List::create();
  bool success = true;
  if (!trained) {
    Rcpp::Rcout << "Training failed: " << model->errMsg() << std::endl;
    success = false;
  }
  // NORMALISE UPFRONT - DIFFERENT THAN ORIGINAL CODE 
  // - original code dumps data to disk, next imports it and during import normalisation happens after which we can do nearest calculations
  // - the R wrapper only writes to disk at request so we need to normalise upfront in order to do directly nearest calculations
  if (normalize) {
    //Rcpp::Rcout << "Finished training: finalising with embedding normalisation" << std::endl;
    model->normalize();
  }
  
  // Return model + model information
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("data") = Rcpp::List::create(
      //Rcpp::Named("file") = trainFile,
      //Rcpp::Named("stopwords") = stopWordsFile,
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
      Rcpp::Named("expValueMax") = expValueMax
      //Rcpp::Named("split_words") = wordDelimiterChars,
      //Rcpp::Named("split_sents") = endOfSentenceChars
    )
  );
  out.attr("class") = "word2vec_trained";
  return out;
}

/*
// [[Rcpp::export]]
Rcpp::List w2v_load_model(std::string file, bool normalize = true) {
  bool normalise = normalize;
  Rcpp::XPtr<w2v::w2vModel_t> model(new w2v::w2vModel_t(), true);
  if (!model->load(file, normalize = normalise)) {
    Rcpp::stop(model->errMsg());
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model") = model,
    Rcpp::Named("model_path") = file,
    Rcpp::Named("dim") = model->vectorSize(),
    Rcpp::Named("vocabulary") = model->modelSize()
  );
  out.attr("class") = "word2vec";
  return out;
}

// [[Rcpp::export]]
bool w2v_save_model(SEXP ptr, std::string file) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  bool success = model->save(file);
  return success;
}
*/

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
    Rcpp::Named("rank") = rank,
    Rcpp::Named("stringsAsFactors") = false
  );
  return out;
}


// [[Rcpp::export]]
Rcpp::List w2v_nearest_vector(SEXP ptr, 
                              const std::vector<float> &x, 
                              std::size_t top_n = 10,
                              float min_distance = 0.0) {
  Rcpp::XPtr<w2v::w2vModel_t> model(ptr);
  w2v::vector_t *vec = new w2v::vector_t(x);
  
  std::vector<std::pair<std::string, float>> nearest;
  model->nearest(*vec, nearest, top_n, min_distance);
  
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
    Rcpp::Named("term") = keys,
    Rcpp::Named("similarity") = distance,
    Rcpp::Named("rank") = rank,
    Rcpp::Named("stringsAsFactors") = false
  );
  return out;
}

/* NOTE: temporarily disabled
 
// [[Rcpp::export]]
Rcpp::NumericMatrix w2v_read_binary(const std::string modelFile, bool normalize, std::size_t n) {
  try {
    const std::string wrongFormatErrMsg = "model: wrong model file format";
    
    // map model file, exception will be thrown on empty file
    w2v::fileMapper_t input(modelFile);
    
    // parse header
    off_t offset = 0;
    // get words number
    std::string nwStr;
    char ch = 0;
    while ((ch = (*(input.data() + offset))) != ' ') {
      nwStr += ch;
      if (++offset >= input.size()) {
        throw std::runtime_error(wrongFormatErrMsg);
      }
    }
    
    // get vector size
    offset++; // skip ' ' char
    std::string vsStr;
    while ((ch = (*(input.data() + offset))) != '\n') {
      vsStr += ch;
      if (++offset >= input.size()) {
        throw std::runtime_error(wrongFormatErrMsg);
      }
    }
    
    std::size_t m_mapSize;
    uint16_t m_vectorSize;
    try {
      m_mapSize = static_cast<std::size_t>(std::stoll(nwStr));
      m_vectorSize = static_cast<uint16_t>(std::stoi(vsStr));
    } catch (...) {
      throw std::runtime_error(wrongFormatErrMsg);
    }
    if(m_mapSize > n){
      m_mapSize = n;
    }
    Rcpp::NumericMatrix embedding(m_mapSize, m_vectorSize);
    Rcpp::StringVector embedding_words(m_mapSize);
    //std::fill(embedding.begin(), embedding.end(), Rcpp::NumericVector::get_na());
    
    // get pairs of word and vector
    offset++; // skip last '\n' char
    std::string word;
    for (std::size_t i = 0; i < m_mapSize; ++i) {
      // get word
      word.clear();
      while ((ch = (*(input.data() + offset))) != ' ') {
        if (ch != '\n') {
          word += ch;
        }
        // move to the next char and check boundaries
        if (++offset >= input.size()) {
          throw std::runtime_error(wrongFormatErrMsg);
        }
      }
      embedding_words[i] = word;
      
      // skip last ' ' char and check boundaries
      if (static_cast<off_t>(++offset + m_vectorSize * sizeof(float)) > input.size()) {
        throw std::runtime_error(wrongFormatErrMsg);
      }
      
      // get word's vector
      std::vector<float> v(m_vectorSize);
      std::memcpy(v.data(), input.data() + offset, m_vectorSize * sizeof(float));
      offset += m_vectorSize * sizeof(float); // vector size
      
      if(normalize){
        // normalize vector
        float med = 0.0f;
        for (auto const &j:v) {
          med += j * j;
        }
        if (med <= 0.0f) {
          throw std::runtime_error("failed to normalize vectors");
        }
        med = std::sqrt(med / v.size());
        for (auto &j:v) {
          j /= med;
        }
      }
      for(unsigned int j = 0; j < v.size(); j++){
        //embedding(i, j) = (float)((*v)[j]);
        embedding(i, j) = v[j];
      }
      
    }
    rownames(embedding) = embedding_words;
    return embedding;
  } catch (const std::exception &_e) {
    std::string m_errMsg = _e.what();
  } catch (...) {
    std::string m_errMsg = "model: unknown error";
  }
  Rcpp::NumericMatrix embedding_default;
  return embedding_default;
}

// [[Rcpp::export]]
Rcpp::List d2vec(SEXP ptr, Rcpp::StringVector x, std::string wordDelimiterChars = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r") {
  Rcpp::XPtr<w2v::w2vModel_t> model_w2v(ptr);
  Rcpp::XPtr<w2v::d2vModel_t> model_d2v(new w2v::d2vModel_t(model_w2v->vectorSize()), true);
  for (int i = 0; i < x.size(); i++){
    std::string input = Rcpp::as<std::string>(x[i]);
    w2v::doc2vec_t doc2vec(model_w2v, input, wordDelimiterChars);
    model_d2v->set(i+1, doc2vec);
  }
  Rcpp::List out = Rcpp::List::create(
    Rcpp::Named("model_d2v") = model_d2v,
    Rcpp::Named("model_w2v") = model_w2v,
    Rcpp::Named("dim") = model_w2v->vectorSize()
  );
  out.attr("class") = "doc2vec";
  return out;
}



// [[Rcpp::export]]
Rcpp::DataFrame d2vec_nearest(SEXP ptr_w2v, SEXP ptr_d2v, Rcpp::StringVector x, 
                              std::string wordDelimiterChars = " \n,.-!?:;/\"#$%&'()*+<=>@[]\\^_`{|}~\t\v\f\r") {
  std::vector<std::string> doc_ids =  x.attr("names");
  std::string text = Rcpp::as<std::string>(x[0]);
  std::string doc_id = doc_ids[0];
  
  Rcpp::XPtr<w2v::w2vModel_t> model_w2v(ptr_w2v);
  Rcpp::XPtr<w2v::d2vModel_t> model_d2v(ptr_d2v);
  w2v::doc2vec_t doc2vec(model_w2v, text, wordDelimiterChars);
  
  // get nearest article IDs from the model
  std::vector<std::pair<std::size_t, float>> nearest;
  model_d2v->nearest(doc2vec, nearest, model_d2v->modelSize());
  
  std::vector<int> keys;
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
    Rcpp::Named("term1") = doc_id,
    Rcpp::Named("term2") = keys,
    Rcpp::Named("similarity") = distance,
    Rcpp::Named("rank") = rank,
    Rcpp::Named("stringsAsFactors") = false
  );
  return out;
}

 */
