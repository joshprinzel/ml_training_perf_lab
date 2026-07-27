#include <torch/extension.h>
#include <torch/library.h>

TORCH_LIBRARY(ml_training_perf, library) {
    library.def(
        "scaled_relu(Tensor input, float scale) -> Tensor"
    );
}

// The module can be imported as ml_training_perf._C.
// Operator access occurs through torch.ops and the dispatcher rather than
// through manually exposed pybind functions.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    
}