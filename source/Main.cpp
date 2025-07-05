/* NiuTrans.NMT - an open-source neural machine translation system.
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
 * $Created by: Chi Hu (huchinlp@gmail.com) 2021-11-06
 */

#include <iostream>
#include <time.h>
#include "./s2t/S2TConfig.h"
#include "./nmt/train/Trainer.h"
#include "./nmt/translate/Translator.h"
#include "./s2t/S2TModel.h"
#include "./s2t/generate/Generator.h"
#include "./s2t/S2TVocab.h"
#include "niutensor/tensor/function/GELU.h"

#include "./s2t/WaveLoader.h"
#include "./s2t/FeatureWindow.h"
#include "./s2t/Fbank.h"

using namespace nmt;
using namespace s2t;
using namespace nts;

int main(int argc, const char** argv)
{

    // std::ios_base::sync_with_stdio(false);
    // std::cin.tie(NULL);


    if (argc == 0)
        return 1;

    DISABLE_GRAD;

    /* load configurations */
    // S2TConfig config(argc, argv);
    S2TConfig* config = new S2TConfig(argc, argv);
    // S2TModel model;
    S2TModel *model = new S2TModel();


    model->InitModel(config);

    Generator generator;

    CheckNTErrors(!(config->model.NAR && strlen(config->inference.inputFN) > 0),
    "NAR is not implemented for offline decoding.");
    CheckNTErrors(!(config->model.NAR && (config->inference.beamSize!=1)),
    "NAR is not implemented for beam search > 1.");
    CheckNTErrors(strcmp(config->inference.inputFN, "") || strcmp(config->extractor.inputAudio,""),
        "Giving input path to choose offline or input audio to choose online decoding");
    // Choosing online inference with speech extractor
    if (strlen(config->extractor.inputAudio) != 0)
    {
        generator.Init(config, model, false);
    }
    // Choosing offline inference with batch decoding
    else if (strlen(config->inference.inputFN) != 0)
    {
        generator.Init(config, model);
    }
    else
    {
        CheckNTErrors((strlen(config->inference.inputFN) != 0 || strlen(config->extractor.inputAudio) != 0),
            "Giving input path to choose offline or input audio to choose online decoding");
    }
    clock_t start, finish;
    double duration;
    start = clock();
    generator.Generate();
    finish = clock();
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    std::cout << "Time: " << duration << endl;
    return 0;
}
