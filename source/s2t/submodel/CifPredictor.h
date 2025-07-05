#ifndef __Predictor_H__
#define __Predictor_H__

#include "../S2TConfig.h"
#include "../../niutensor/tensor/XTensor.h"
#include "../../nmt/submodel/LinearLayer.h"
#include "../../niutensor/tensor/function/Rectify.h"
#include "../../niutensor/tensor/function/Sigmoid.h"

using namespace nts;
/* the s2t namespace */
namespace s2t
{

/* a fnn: y = max(0, x * w1 + b1) * w2 + b2 */
class CifPredictor
{
public:
    /* indicates whether train the model */
    bool isTraining;

    /* device id */
    int devID;

    /* size of input vector */
    int inSize;

    /* size of output vector */
    int hSize;

    /* number of convolution layer */
    int nConvNAR;

    DTYPE smooth_factor;

    DTYPE threshold;

    DTYPE noise_threshold;

    DTYPE tail_threshold;

    /* kernel sizes of convolution layer */
    vector<int> convKernelsNAR;

    /* stride sizes of convolution layer */
    vector<int> convStridesNAR;

    /* matrix of kernel tensor */
    XTensor* kernels;

    /* matrix of convolution bias tensor */
    XTensor* biases;

    /* dropout probability */
    DTYPE dropoutP;

    LinearLayer* prelinear;

    /* tail process function*/
    void _tailProcess(XTensor& input, XTensor& alphas, XTensor& token_num, XTensor mask);

    XTensor cif(XTensor& input, XTensor& alphas);

public:
    /* set the training flag */
    void SetTrainingFlag(bool myIsTraining);

    /* constructor */
    CifPredictor();

    /* de-constructor */
    ~CifPredictor();

    /* initialize the model */
    void InitModel(S2TConfig& config);

    /* make the network */
    XTensor Make(XTensor input, XTensor& mask);
};

} /* end of the s2t namespace */

#endif /* __Predictor__ */
