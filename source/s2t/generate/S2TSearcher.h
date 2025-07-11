/* NiuTrans.S2T - an open-source speech to text system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */


 /*
  * $Created by: yuhao zhang(yoohao.zhang@gmail.com) 2023-09-22
  */

#ifndef __S2TSEARCHER_H__
#define __S2TSEARCHER_H__
#include "../S2TModel.h"
#include "../../nmt/translate/Searcher.h"
#include "../S2TVocab.h"

using namespace nmt;


namespace s2t {

    class S2TGreedySearch : public GreedySearch
    {
    private:
        /* max length of the generated sequence */
        int maxLen;

        /* batch size */
        int batchSize;

        /* array of the end symbols */
        int* endSymbols;

        /* number of the end symbols */
        int endSymbolNum;

        /* array of the start symbols */
        int* startSymbols;

        /* number of the start symbols */
        int startSymbolNum;

        /* array of the suppress symbols */
        int* suppressSymbols;

        /* number of the suppress symbols */
        int suppressSymbolNum;

        /* scalar of the input sequence (for max number of search steps) */
        float scalarMaxLength;

        /* target language vocab */
        S2TVocab* vocab;

        bool withoutTimeStamps;


    public:

        S2TGreedySearch();

        ~S2TGreedySearch();
        /* initialize the model */
        void Init(S2TConfig& config, S2TVocab* tgtVocab);

        /* initialize the start symbols */
        void InitStartSymbols(S2TConfig& config);

        /* initialize the prompt symbols */
        void InitPromptSymbols();

        void InitPromptSymbols(int* textPrompt, const int textPromptNum, int* decodingPrompt, const int decodingPromptNum);

        /* initialize the suppress symbols */
        void InitSuppressSymbols(S2TConfig& config, int* tokens=NULL, const int num=0);

        /* check if the token is an end symbol */
        bool IsEnd(int token);

        /* search for the most promising states */
        void Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs);

        /* search for the most promising states using non-autoregressive decoding, isNar SHOULD BE TRUE*/
        void Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs, int audioLength);

        XTensor Suppress(IntList** tokens, XTensor& input, bool isBegin=FALSE);

        XTensor Predict(XTensor& tokens, XTensor& logits, XTensor* sumLogprobs=NULL);

    };

    class S2TPredictor : public Predictor
    {
    private:

        /* beam size */
        int beamSize;

        /* batch size */
        int batchSize;
        
        /* pointer to the transformer model */
        S2TModel* m;

        /* current state */
        StateBundle* s;

        /* array of the end symbols */
        int* endSymbols;

        /* number of the end symbols */
        int endSymbolNum;

        /* start symbol */
        int* startSymbols;

        /* number of the start symbols */
        int startSymbolNum;

        /* suppress symbol */
        int* suppressSymbols;

        /* number of the suppress symbols */
        int suppressSymbolNum;

        /* target language vocab */
        S2TVocab* vocab;

        bool withoutTimeStamps;

    public:

        S2TPredictor();

        ~S2TPredictor();

        void Init(int batchN, int beamN, int* endS, int endN, int* startS, int startN, int* suppS, int suppN, S2TVocab* tgtVocab, bool noTimeStamps=true);

        void Read(S2TModel* model, StateBundle* state);

        /* create an initial state */
        void Create(const XTensor* input, int beamSize, StateBundle* state);

        void Predict(StateBundle* next, XTensor& aliveState, XTensor& encoding,
            XTensor& inputEnc, XTensor& paddingEnc, int batchSize, bool isStart,
            XTensor& reorderState, bool needReorder, int nstep, IntList** outputs);

        XTensor Suppress(StateBundle* cur, XTensor& input, bool isBegin=FALSE);

    };

    class S2TBeamSearch : public BeamSearch
    {
    private:
        /* the alpha parameter controls the length preference */
        float alpha;

        /* predictor */
        S2TPredictor predictor;

        /* max length of the generated sequence */
        int maxLen;

        /* beam size */
        int beamSize;

        /* batch size */
        int batchSize;

        /* we keep the final hypotheses in a heap for each sentence in the batch. */
        XHeap<MIN_HEAP, float>* fullHypos;

        /* array of the end symbols */
        int* endSymbols;

        /* number of the end symbols */
        int endSymbolNum;

        /* start symbol */
        int* startSymbols;

        /* number of the start symbols */
        int startSymbolNum;

        /* suppress symbol */
        int* suppressSymbols;

        /* number of the suppress symbols */
        int suppressSymbolNum;

        /* scalar of the input sequence (for max number of search steps) */
        float scalarMaxLength;

        /* indicate whether the early stop strategy is used */
        bool isEarlyStop;

        /* pids for alive states */
        IntList aliveStatePids;

        /* alive sentences */
        IntList aliveSentList;

        /* whether we need to reorder the states */
        bool needReorder;

        /* target language vocab */
        S2TVocab* vocab;

        bool withoutTimeStamps;

    public:

        /*TODO*/
        /* constructor */
        S2TBeamSearch();

        /* de-constructor */
        ~S2TBeamSearch();

        /* initialize the model */
        void Init(S2TConfig& config, S2TVocab* tgtVocab);

        /* initialize the start symbols */
        void InitStartSymbols(S2TConfig& config);

        /* initialize the prompt symbols */
        void InitPromptSymbols();

        /* initialize the suppress symbols */
        void InitSuppressSymbols(S2TConfig& config, int* tokens = NULL, const int num = 0);

        /* preparation */
        void Prepare(int myBatchSize, int myBeamSize);

        /* compute the model score for each hypotheses */
        void Score(StateBundle* prev, StateBundle* beam);

        /* generate token indices via beam pruning */
        // void Generate(StateBundle* prev, StateBundle* beam);

        /* expand the search graph */
        void Expand(StateBundle* prev, StateBundle* beam, XTensor& reorderState);

        /* collect hypotheses with ending symbol */
        void Collect(StateBundle* beam);

        /* fill the hypotheses heap with incomplete hypotheses */
        void FillHeap(StateBundle* beam);

        /* save the output sequences and score */
        void Dump(IntList** output, XTensor* score);

        /* check if the token is an end symbol */
        bool IsEnd(int token);  

        /* search for the most promising states */
        void Search(S2TModel* model, XTensor& input, XTensor& padding, IntList** outputs, XTensor& score);
    };


} /* end of s2t namespace */


#endif
