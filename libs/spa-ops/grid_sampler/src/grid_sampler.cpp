#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA Tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

void launch_grid_sampler_2d_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid,
    int64_t padding_mode, bool align_corners);

void launch_grid_sampler_3d_forward_kernel(
    const torch::TensorBase &output, const torch::TensorBase &input, const torch::TensorBase &grid,
    int64_t padding_mode, bool align_corners);

void launch_grid_sampler_2d_backward_kernel(
    const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid,
    const torch::TensorBase &grad_output, const torch::TensorBase &input,
    const torch::TensorBase &grid, int64_t padding_mode, bool align_corners);

void launch_grid_sampler_3d_backward_kernel(
    const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid,
    const torch::TensorBase &grad_output, const torch::TensorBase &input,
    const torch::TensorBase &grid, int64_t padding_mode, bool align_corners);

void launch_grid_sampler_2d_backward_backward_kernel(
    const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid, const torch::TensorBase &grad_grad_output,
    const torch::TensorBase &grad2_grad_input, const torch::TensorBase &grad2_grad_grid,
    const torch::TensorBase &input, const torch::TensorBase &grid, const torch::TensorBase &grad_output,
    int64_t padding_mode, bool align_corners);

void launch_grid_sampler_3d_backward_backward_kernel(
  const torch::TensorBase &grad_input, const torch::TensorBase &grad_grid, const torch::TensorBase &grad_grad_output,
  const torch::TensorBase &grad2_grad_input, const torch::TensorBase &grad2_grad_grid,
  const torch::TensorBase &input, const torch::TensorBase &grid, const torch::TensorBase &grad_output,
  int64_t padding_mode, bool align_corners);

torch::Tensor grid_sampler_2d_forward(const torch::Tensor& input, const torch::Tensor& grid,
                                   int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = at::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2]}, input.options());
  launch_grid_sampler_2d_forward_kernel(
      output, input, grid, padding_mode, align_corners);
  return output;
}

torch::Tensor grid_sampler_3d_forward(const torch::Tensor &input, const torch::Tensor &grid,
                                   int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  auto in_size = input.sizes();
  auto grid_size = grid.sizes();
  auto output = torch::empty(
      {in_size[0], in_size[1], grid_size[1], grid_size[2], grid_size[3]}, input.options());
  launch_grid_sampler_3d_forward_kernel(
      output, input, grid, padding_mode, align_corners);
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
grid_sampler_2d_backward(const torch::Tensor &grad_output, const torch::Tensor &input,
                         const torch::Tensor &grid, int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(grad_output)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, padding_mode, align_corners);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor>
grid_sampler_3d_backward(const torch::Tensor &grad_output, const torch::Tensor &input,
                         const torch::Tensor &grid, int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(grad_output)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_3d_backward_kernel(
      grad_input, grad_grid, grad_output, input,
      grid, padding_mode, align_corners);
  return std::make_tuple(grad_input, grad_grid);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
grid_sampler_2d_backward_backward(const torch::Tensor &grad2_grad_input, const torch::Tensor &grad2_grad_grid,
                                  const torch::Tensor &input, const torch::Tensor &grid, const torch::Tensor &grad_output,
                                  int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(grad2_grad_input)
  CHECK_INPUT(grad2_grad_grid)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(grad_output)
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grad_output = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_2d_backward_backward_kernel(grad_input, grad_grid, grad_grad_output, grad2_grad_input, grad2_grad_grid,
                                                  input, grid, grad_output, padding_mode, align_corners);
  return std::make_tuple(grad_input, grad_grid, grad_grad_output);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> 
grid_sampler_3d_backward_backward(const torch::Tensor &grad2_grad_input, const torch::Tensor &grad2_grad_grid,
                                  const torch::Tensor &input, const torch::Tensor &grid, const torch::Tensor &grad_output,
                                  int64_t padding_mode, bool align_corners) {
  CHECK_INPUT(grad2_grad_input)
  CHECK_INPUT(grad2_grad_grid)
  CHECK_INPUT(input)
  CHECK_INPUT(grid)
  CHECK_INPUT(grad_output)
  auto grad_input = torch::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grid = torch::empty_like(grid, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto grad_grad_output = torch::zeros_like(grad_output, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  launch_grid_sampler_3d_backward_backward_kernel(grad_input, grad_grid, grad_grad_output, grad2_grad_input, grad2_grad_grid,
                                                  input, grid, grad_output, padding_mode, align_corners);
  return std::make_tuple(grad_input, grad_grid, grad_grad_output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("grid_sampler_3d_forward", &grid_sampler_3d_forward, "grid_sampler_3d_forward");
  m.def("grid_sampler_3d_backward", &grid_sampler_3d_backward, "grid_sampler_3d_backward");
  m.def("grid_sampler_3d_backward_backward", &grid_sampler_3d_backward_backward, "grid_sampler_3d_backward_backward");
  m.def("grid_sampler_2d_forward", &grid_sampler_2d_forward, "grid_sampler_2d_forward");
  m.def("grid_sampler_2d_backward", &grid_sampler_2d_backward, "grid_sampler_2d_backward");
  m.def("grid_sampler_2d_backward_backward", &grid_sampler_2d_backward_backward, "grid_sampler_2d_backward_backward");
}