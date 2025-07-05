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

#ifndef __ADAPTER_H__
#define __ADAPTER_H__

#include "../../nmt/Encoder.h"
#include "../S2TConfig.h"
using namespace nmt;
namespace s2t
{
class Adapter : public AttEncoder
{
public:

    /* constructor */
    Adapter();

    /* de-constructor */
    ~Adapter();

    /* initialize the model */
    void InitModel(S2TConfig& config);

    /* initialize the model */
    XTensor applyLfr(XTensor& input);

    /* run encoding for inference with post-norm */
    XTensor RunFastPreNorm(XTensor& input, XTensor* mask);
};
}
#endif