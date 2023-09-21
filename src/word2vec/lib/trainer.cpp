#include <Rcpp.h>
/**
 * @file
 * @brief trainer class of word2vec model
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainer.hpp"

namespace w2v {
    trainer_t::trainer_t(const std::shared_ptr<trainSettings_t> &_trainSettings,
                         const std::shared_ptr<vocabulary_t> &_vocabulary,
                         const std::shared_ptr<corpus_t> &_corpus,
                         const std::shared_ptr<fileMapper_t> &_fileMapper, // NOTE: remove
                         std::function<void(float, float)> _progressCallback): m_threads() {
        trainThread_t::sharedData_t sharedData;

        if (!_trainSettings) {
            throw std::runtime_error("train settings are not initialized");
        }
        sharedData.trainSettings = _trainSettings;

        if (!_vocabulary) {
            throw std::runtime_error("vocabulary object is not initialized");
        }
        sharedData.vocabulary = _vocabulary;

        if (!_corpus && !_fileMapper) {
            throw std::runtime_error("corpus and file mapper objects are not initialized");
        }
        sharedData.corpus = _corpus;
        sharedData.fileMapper = _fileMapper;
        
        sharedData.bpWeights.reset(new std::vector<float>(_trainSettings->size * _vocabulary->size(), 0.0f));
        sharedData.expTable.reset(new std::vector<float>(_trainSettings->expTableSize));
        for (uint16_t i = 0; i < _trainSettings->expTableSize; ++i) {
            // Precompute the exp() table
            (*sharedData.expTable)[i] =
                    exp((i / static_cast<float>(_trainSettings->expTableSize) * 2.0f - 1.0f)
                                           * _trainSettings->expValueMax);
            // Precompute f(x) = x / (x + 1)
            (*sharedData.expTable)[i] = (*sharedData.expTable)[i] / ((*sharedData.expTable)[i] + 1.0f);
        }
        
        if (_trainSettings->withHS) {
            std::vector<std::size_t> frequencies;
            _vocabulary->frequencies(frequencies);
            sharedData.huffmanTree.reset(new huffmanTree_t(frequencies));;
        }

        if (_progressCallback != nullptr) {
            sharedData.progressCallback = _progressCallback;
        }

        sharedData.processedWords.reset(new std::atomic<std::size_t>(0));
        sharedData.alpha.reset(new std::atomic<float>(_trainSettings->alpha));
        
        // if (_corpus) {
        //     // NOTE : corpus has no sentence delimiter
        //     m_matrixSize = sharedData.trainSettings->size * sharedData.vocabulary->size() + 100;
        // } else {
            m_matrixSize = sharedData.trainSettings->size * sharedData.vocabulary->size();
        //}
        //Rcpp::Rcout << "corpus->texts.size(): " << sharedData.corpus->texts.size() << "\n";
        //Rcpp::Rcout << "vocabulary->size(): " << sharedData.vocabulary->size() << "\n";
        //Rcpp::Rcout << "_trainSettings->threads: " << (int)_trainSettings->threads << "\n";
        for (uint8_t i = 0; i < _trainSettings->threads; ++i) {
            // trainThread_t t(i, sharedData);
            // Rcpp::Rcout << "thread: " << (int)i << " from " << t.range.first << " to " << t.range.second << "\n"; 
            m_threads.emplace_back(new trainThread_t(i, sharedData));
        }
        //throw std::runtime_error("m_threads.emplace_back()");
    }

    void trainer_t::operator()(std::vector<float> &_trainMatrix) noexcept {
        // input matrix initialized with small random values
        std::random_device randomDevice;
        std::mt19937_64 randomGenerator(randomDevice());
        std::uniform_real_distribution<float> rndMatrixInitializer(-0.005f, 0.005f);
        _trainMatrix.resize(m_matrixSize);
        std::generate(_trainMatrix.begin(), _trainMatrix.end(), [&]() {
            float v = (float)(Rcpp::runif(1, -0.005f, 0.005f)[0]);
            return v;
            //return rndMatrixInitializer(randomGenerator);
        });

        for (auto &i:m_threads) {
            i->launch(_trainMatrix);
        }

        for (auto &i:m_threads) {
            i->join();
        }
    }
}
