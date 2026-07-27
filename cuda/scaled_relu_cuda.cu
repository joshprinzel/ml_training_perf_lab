#include "ml_training_perf/scaled_relu.hpp"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace ml_training_perf{
    namespace{
        constexpr int kThreadsPerBlock = 256;

        __global__ void scaled_relu_kernel(
            const float* __restrict__ input,
            float* __restrict__ output,
            const std::int64_t element_count,
            const float scale
        ){
            const std::int64_t index = 
                static_cast<std::int64_t>(blockIdx.x) *
                static_cast<std::int64_t>(blockDim.x) + 
                static_cast<std::int64_t>(threadIdx.x);

            if(index >= element_count){
                return;
            }

            const float value = input[index];

            if(isnan(value)){
                output[index] = value;
            }else{
                output[index] = value > 0.0F ? value * scale : 0.0F;
            }


        }
    } //namespace

    at::Tensor scaled_relu_cuda(
        const at::Tensor& input,
        const double scale
    ){
        TORCH_CHECK(
            input.is_cuda(),
            "scaled_relu: expected a CUDA tensor"
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

        // Set the CUDA device associated with input for this scope and restore
        // the previous device when the guard is destroyed
        const c10::cuda::CUDAGuard device_guard(input.device());

        at::Tensor output = at::empty_like(input);

        const std::int64_t element_count = input.numel();
        if(element_count == 0){
            return output;
        }

        const std::int64_t block_count = (element_count + kThreadsPerBlock - 1) / kThreadsPerBlock;

        cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        scaled_relu_kernel<<<
                static_cast<unsigned int>(block_count),
                kThreadsPerBlock,
                0,
                stream
            >>>(
                input.data_ptr<float>(),
                output.data_ptr<float>(),
                element_count,
                static_cast<float>(scale)
            );

        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return output;
    }


} //ml_training_perf

TORCH_LIBRARY_IMPL(ml_training_perf, CUDA, library){
    library.impl(
        "scaled_relu",
        TORCH_FN(ml_training_perf::scaled_relu_cuda)
    );
}