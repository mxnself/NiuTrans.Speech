/* NiuTrans.S2T - an open-source speech-to-text system.
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
  * $Created by: Yuhao Zhang (yoohao.zhang@gmail.com) 2023-09-19
  */

 /*
 This class generate test speechs with a trained model.
 It will dump the result to the output file if specified, else the standard output.
 */

#ifndef __TRANSLATOR_S2T__
#define __TRANSLATOR_S2T__

#include "../Fbank.h"
#include "../FeatureWindow.h"
#include "../S2TModel.h"
#include "../../nmt/translate/Searcher.h"
#include "S2TSearcher.h"
#include "S2TGeneratorDataset.h"
#include "../S2TVocab.h"

 /* the s2t namespace */
namespace s2t
{

    class Generator
    {

    private:
        /* the translation model */
        S2TModel* model;

        /* for batching */
        S2TGeneratorDataset* batchLoader;

        /* the searcher for translation */
        void* searcher;

        /* configuration of the S2T system */
        S2TConfig* config;

        /* target language vocab */
        S2TVocab* vocab;

        OfflineFeatureTpl<FbankComputer>* oft;

    private:
        /* translate a batch of sequences */
        XTensor DecodingBatch(XTensor& batchEnc, XTensor& paddingEnc, IntList& indices, int audio_length=0);



    public:
        /* constructor */
        Generator();

        /* de-constructor */
        ~Generator();

        /* initialize the translator */
        void Init(S2TConfig* myConfig, S2TModel* myModel, bool offline = true);

        /* the generate function */
        bool Generate();
    };

} /* end of the s2t namespace */

#endif /*  */
