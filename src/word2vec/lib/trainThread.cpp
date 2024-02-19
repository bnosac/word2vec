/**
 * @file
 * @brief trainThread trains a word2vec model from the specified part of train data set file
 * @author Max Fomichev
 * @date 20.12.2016
 * @copyright Apache License v.2 (http://www.apache.org/licenses/LICENSE-2.0)
*/

#include "trainThread.hpp"

namespace w2v {
    trainThread_t::trainThread_t(uint8_t _id, const sharedData_t &_sharedData) :
            m_sharedData(_sharedData), m_randomDevice(), m_randomGenerator(m_randomDevice()),
            m_rndWindowShift(0, static_cast<short>((m_sharedData.trainSettings->window - 1))),
            m_downSampling(), m_nsDistribution(), m_hiddenLayerVals(), m_hiddenLayerErrors(),
            m_thread() {

        if (!m_sharedData.trainSettings) {
            throw std::runtime_error("train settings are not initialized");
        }

        if (m_sharedData.trainSettings->sample > 0.0f) {
            m_downSampling.reset(new downSampling_t(m_sharedData.trainSettings->sample,
                                                    m_sharedData.corpus->trainWords));
        }

        if (m_sharedData.trainSettings->negative > 0) {
            m_nsDistribution.reset(new nsDistribution_t(m_sharedData.corpus->frequency));
        }

        if (m_sharedData.trainSettings->withHS && !m_sharedData.huffmanTree) {
            throw std::runtime_error("Huffman tree object is not initialized");
        }

        m_hiddenLayerErrors.reset(new std::vector<float>(m_sharedData.trainSettings->size));
        if (!m_sharedData.trainSettings->withSG) {
            m_hiddenLayerVals.reset(new std::vector<float>(m_sharedData.trainSettings->size));
        }

        if (!m_sharedData.corpus) {
            throw std::runtime_error("corpus object is not initialized");
        }
        
        // NOTE: specify range for workers
        auto n = m_sharedData.corpus->texts.size();
        auto threads = m_sharedData.trainSettings->threads;
        range = std::make_pair(floor((n / threads) * _id),
                               floor((n / threads) * (_id + 1)) - 1);
        
    }

    void trainThread_t::worker(std::vector<float> &_trainMatrix) noexcept {
        
        Rcpp::Rcout << "Texts: " << range.first << " to " << range.second << "\n";        
        for (auto g = m_sharedData.trainSettings->iterations; g > 0; --g) {
            //Rcpp::Rcout << "g: " << (int)g << "\n";
            //bool exitFlag = false;
            std::size_t threadProcessedWords = 0;
            std::size_t prvThreadProcessedWords = 0;

            //std::size_t h = range.first; // NOTE: only used for corpus
            auto wordsPerAllThreads = m_sharedData.trainSettings->iterations
                                      * m_sharedData.corpus->trainWords;
            auto wordsPerAlpha = wordsPerAllThreads / 10000;
            //while (!exitFlag) {
            //while (h <= range.second) {
            for (std::size_t h = range.first; h <= range.second; ++h) {

                // calc alpha
                if (threadProcessedWords - prvThreadProcessedWords > wordsPerAlpha) { // next 0.01% processed
                    *m_sharedData.processedWords += threadProcessedWords - prvThreadProcessedWords;
                    prvThreadProcessedWords = threadProcessedWords;

                    float ratio =
                            static_cast<float>(*(m_sharedData.processedWords)) / wordsPerAllThreads;

                    auto curAlpha = m_sharedData.trainSettings->alpha * (1 - ratio);
                    if (curAlpha < m_sharedData.trainSettings->alpha * 0.0001f) {
                        curAlpha = m_sharedData.trainSettings->alpha * 0.0001f;
                    }
                    (*m_sharedData.alpha) = curAlpha;

                    if (m_sharedData.progressCallback != nullptr) {
                        m_sharedData.progressCallback(curAlpha, ratio * 100.0f);
                    }
                }
                
                // if (h > range.second) {
                //     exitFlag = true; // EOF or end of requested region
                //     break;
                // }
                text_t text = m_sharedData.corpus->texts[h];
                
                // read sentence
                std::vector<unsigned int> sentence;
                sentence.reserve(text.size());
                for (size_t i = 0; i < text.size(); ++i) {

                    auto &word = text[i];
                    if (word == 0) { // padding
                        continue; 
                    }

                    threadProcessedWords++;
                    
                    if (m_sharedData.trainSettings->sample > 0.0f) {
                        if ((*m_downSampling)(m_sharedData.corpus->frequency[word - 1], m_randomGenerator)) {
                            continue; // skip this word
                        }
                    }
                    sentence.push_back(word - 1); // zero-based index of words
                }
                
                if (m_sharedData.trainSettings->withSG) {
                    skipGram(sentence, _trainMatrix);
                } else {
                    cbow(sentence, _trainMatrix);
                }
            }
        }
    }

    inline void trainThread_t::cbow(const std::vector<unsigned int> &_sentence,
                                    std::vector<float> &_trainMatrix) noexcept {
        
        if (_sentence.size() == 0)
            return;
        for (std::size_t i = 0; i < _sentence.size(); ++i) {
            // hidden layers initialized with 0 values
            std::memset(m_hiddenLayerVals->data(), 0, m_hiddenLayerVals->size() * sizeof(float));
            std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));

            auto rndShift = m_rndWindowShift(m_randomGenerator);
            std::size_t cw = 0;
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                if (posRndWindow >= _sentence.size()) {
                    continue;
                }
                for (std::size_t k = 0; k < m_sharedData.trainSettings->size; ++k) {
                    (*m_hiddenLayerVals)[k] += _trainMatrix[k + _sentence[posRndWindow]
                                                           * m_sharedData.trainSettings->size];
                }
                cw++;
            }
            if (cw == 0) {
                continue;
            }
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; j++) {
                (*m_hiddenLayerVals)[j] /= cw;
            }
            
            if (m_sharedData.trainSettings->withHS) {
                hierarchicalSoftmax(_sentence[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            } else {
                negativeSampling(_sentence[i], *m_hiddenLayerErrors, *m_hiddenLayerVals, 0);
            }
            
            // hidden -> in
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                if (posRndWindow >= _sentence.size()) {
                    continue;
                }
                for (std::size_t k = 0; k < m_sharedData.trainSettings->size; ++k) {
                    _trainMatrix[k + _sentence[posRndWindow] * m_sharedData.trainSettings->size]
                            += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::skipGram(const std::vector<unsigned int> &_sentence,
                                        std::vector<float> &_trainMatrix) noexcept {
        if (_sentence.size() == 0)
            return;
        for (std::size_t i = 0; i < _sentence.size(); ++i) {
            auto rndShift = m_rndWindowShift(m_randomGenerator);
            for (auto j = rndShift; j < m_sharedData.trainSettings->window * 2 + 1 - rndShift; ++j) {
                if (j == m_sharedData.trainSettings->window) {
                    continue;
                }

                auto posRndWindow = i - m_sharedData.trainSettings->window + j;
                if (posRndWindow >= _sentence.size()) {
                    continue;
                }
                // shift to the selected word vector in the matrix
                auto shift = _sentence[posRndWindow] * m_sharedData.trainSettings->size;

                // hidden layer initialized with 0 values
                std::memset(m_hiddenLayerErrors->data(), 0, m_hiddenLayerErrors->size() * sizeof(float));

                if (m_sharedData.trainSettings->withHS) {
                    hierarchicalSoftmax(_sentence[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                } else {
                    negativeSampling(_sentence[i], (*m_hiddenLayerErrors), _trainMatrix, shift);
                }

                for (std::size_t k = 0; k < m_sharedData.trainSettings->size; ++k) {
                    _trainMatrix[k + shift] += (*m_hiddenLayerErrors)[k];
                }
            }
        }
    }

    inline void trainThread_t::hierarchicalSoftmax(std::size_t _index,
                                                   std::vector<float> &_hiddenLayer,
                                                   std::vector<float> &_trainLayer,
                                                   std::size_t _trainLayerShift) noexcept {
        auto huffmanData = m_sharedData.huffmanTree->huffmanData(_index);
        for (std::size_t i = 0; i < huffmanData->huffmanCode.size(); ++i) {
            auto l2 = huffmanData->huffmanPoint[i] * m_sharedData.trainSettings->size;
            // Propagate hidden -> output
            float f = 0.0f;
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                f += _trainLayer[j + _trainLayerShift] * (*m_sharedData.bpWeights)[j + l2];
            }
            if (f < -m_sharedData.trainSettings->expValueMax) {
//            f = 0.0f;
                continue; // original approach
            } else if (f > m_sharedData.trainSettings->expValueMax) {
//            f = 1.0f;
                continue; // original approach
            } else {
                f = (*m_sharedData.expTable)[static_cast<std::size_t>((f + m_sharedData.trainSettings->expValueMax)
                                                                      * (m_sharedData.expTable->size()
                                                                         / m_sharedData.trainSettings->expValueMax /
                                                                         2))];
            }

            auto gradientXalpha = (1.0f - static_cast<float>(huffmanData->huffmanCode[i]) - f) * (*m_sharedData.alpha);
            // Propagate errors output -> hidden
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                _hiddenLayer[j] += gradientXalpha * (*m_sharedData.bpWeights)[j + l2];
            }
            // Learn weights hidden -> output
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                (*m_sharedData.bpWeights)[j + l2] += gradientXalpha * _trainLayer[j + _trainLayerShift];
            }
        }
    }

    inline void trainThread_t::negativeSampling(std::size_t _index,
                                                std::vector<float> &_hiddenLayer,
                                                std::vector<float> &_trainLayer,
                                                std::size_t _trainLayerShift) noexcept {
        for (std::size_t i = 0; i < static_cast<std::size_t>(m_sharedData.trainSettings->negative) + 1; ++i) {
            std::size_t target = 0;
            bool label = false;
            if (i == 0) {
                target = _index;
                label = true;
            } else {
                target = (*m_nsDistribution)(m_randomGenerator);
                if (target == _index) {
                    continue;
                }
            }

            auto l2 = target * m_sharedData.trainSettings->size;
            // Propagate hidden -> output
            float f = 0.0f;
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                f += _trainLayer[j + _trainLayerShift] * (*m_sharedData.bpWeights)[j + l2];
            }
            if (f < -m_sharedData.trainSettings->expValueMax) {
                f = 0.0f;  // original approach
//            continue;
            } else if (f > m_sharedData.trainSettings->expValueMax) {
                f = 1.0f;  // original approach
//            continue;
            } else {
                f = (*m_sharedData.expTable)[static_cast<std::size_t>((f + m_sharedData.trainSettings->expValueMax)
                                                                      * (m_sharedData.expTable->size()
                                                                         / m_sharedData.trainSettings->expValueMax /
                                                                         2))];
            }

            auto gradientXalpha = (static_cast<float>(label) - f) * (*m_sharedData.alpha);
            // Propagate errors output -> hidden
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                _hiddenLayer[j] += gradientXalpha * (*m_sharedData.bpWeights)[j + l2];
            }
            // Learn weights hidden -> output
            for (std::size_t j = 0; j < m_sharedData.trainSettings->size; ++j) {
                (*m_sharedData.bpWeights)[j + l2] += gradientXalpha * _trainLayer[j + _trainLayerShift];
            }
        }
    }
}
