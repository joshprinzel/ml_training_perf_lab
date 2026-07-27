#pragma once

#include <ATen/Tensor.h>

namespace ml_training_perf {


    at::Tensor scaled_relu_cpu(
        const at::Tensor& input,
        double scale
    );

    at::Tensor scaled_relu_cuda(
        const at::Tensor& input,
        double scale
    );



} // namespace ml_training_perf

