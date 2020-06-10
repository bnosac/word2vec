#include <Rcpp.h>
#include <cassert>
#include <string>
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <functional>
#include <cmath>
#include "word2vec.hpp"
#include "wordReader.hpp"
#include "vocabulary.hpp"
#include "trainer.hpp"

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

