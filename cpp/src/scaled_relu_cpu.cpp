#include "ml_training_perf/scaled_relu.hpp"

#include <ATen/ATen.h>
#include <torch/library.h>

#include <cmath>
#include <cstdint>

namespace ml_training_perf{
    at::Tensor scaled_relu_cpu(
        const at::Tensor& input,
        const double scale
    ){
        TORCH_CHECK(
            input.device().is_cpu(),
            "scaled_relu: expected a CPU tensor"
        );

        TORCH_CHECK(
            input.layout() == c10::kStrided,
            "scaled_relu: expected a strided tensor"
        );

        TORCH_CHECK(
            input.scalar_type() == at::kFloat,
            "scaled_relu: expected dtype torch.float32"
        );

        TORCH_CHECK(
            input.is_contiguous(),
            "scaled_relu: expected a contiguous tensor"
        );

        TORCH_CHECK(
            std::isfinite(scale),
            "scaled_relu: scale must be finite"
        );

        at::Tensor output = at::empty_like(input);

        const float* input_data = input.data_ptr<float>();
        float* output_data = output.data_ptr<float>();

        const std::int64_t element_count = input.numel();
        const float scale_value = static_cast<float>(scale);

        for(std::int64_t index = 0; index < element_count; index++){
            const float value = input_data[index];

            if(std::isnan(value)){
                output_data[index] = value;
            }else{
                output_data[index] = value > 0.0F ? value * scale_value : 0.0F;
            }
        }

        return output;
    }
} // namespace ml_training_perf

TORCH_LIBRARY_IMPL(ml_training_perf, CPU, library){
    library.impl(
        "scaled_relu",
        TORCH_FN(ml_training_perf::scaled_relu_cpu)
    );
}