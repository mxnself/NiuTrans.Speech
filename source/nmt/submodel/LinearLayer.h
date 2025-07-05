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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#ifndef __LINEARLAYER_H__
#define __LINEARLAYER_H__

#include "../Config.h"
#include "../../niutensor/tensor/XTensor.h"

using namespace nts;

/* the nmt namespace */
namespace nmt
{


/* a fnn: y = max(0, x * w1 + b1) * w2 + b2 */
class LinearLayer
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* size of input vector */
    int inSize;

    /* size of output vector */
    int outSize;

    /* matrix of transformation 1 */
    XTensor w1;

    /* bias of transformation 1 */
    XTensor b1;


public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    LinearLayer();

    /* de-constructor */
    ~LinearLayer();

    /* initialize the model */
    void InitModel(NMTConfig& config, bool isEnc);

    /* initialize the model */
    void InitModel(NMTConfig& config, int indim, int outdim);

    /* make the network */
    XTensor Make(XTensor& input, bool normalize);
    XTensor Make(XTensor& input);
};

} /* end of the nmt namespace */

#endif /* __FFN_H__ */
