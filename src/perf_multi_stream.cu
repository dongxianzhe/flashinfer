#include <iostream>
#include <thrust/device_vector.h>

#include <cupti.h>

#include"multistream_scheduler.cuh"
#include "flashinfer_ops.cuh"

using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

struct single_decode_input_data {
  size_t seq_len = 8192;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool cooperative = true;

  thrust::device_vector<half>* Q = nullptr;
  thrust::device_vector<half>* K = nullptr;
  thrust::device_vector<half>* V = nullptr;
  thrust::device_vector<half>* O = nullptr;
  thrust::device_vector<half>* tmp = nullptr;
  // Allocate input data:
  single_decode_input_data() {
    Q = new thrust::device_vector<half>(num_qo_heads * head_dim);
    K = new thrust::device_vector<half>(seq_len * num_kv_heads * head_dim);
    V = new thrust::device_vector<half>(seq_len * num_kv_heads * head_dim);
    O = new thrust::device_vector<half>(num_qo_heads * head_dim);
    tmp = new thrust::device_vector<half>(16 * 1024 * 1024);
  }

  ~single_decode_input_data() {
    delete tmp;
    delete O;
    delete V;
    delete K;
    delete Q;
  }
};

struct single_prefill_input_data {
  size_t kv_len = 8192;
  size_t qo_len = kv_len;
  size_t num_kv_heads = 32;
  size_t num_qo_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool causal = false;
  bool cooperative = true;
  bool allow_fp16_qk_reduction = false;

  // Allocate input data:
  thrust::device_vector<half>* Q = nullptr;
  thrust::device_vector<half>* K = nullptr;
  thrust::device_vector<half>* V = nullptr;
  thrust::device_vector<uint8_t>* mask = nullptr;
  thrust::device_vector<half>* O = nullptr;
  thrust::device_vector<half>* tmp = nullptr;

  single_prefill_input_data() {
    Q = new thrust::device_vector<half>(qo_len * num_qo_heads * head_dim);
    K = new thrust::device_vector<half>(kv_len * num_kv_heads * head_dim);
    V = new thrust::device_vector<half>(kv_len * num_kv_heads * head_dim);
    mask = new thrust::device_vector<uint8_t>(qo_len * kv_len / 8);
    O = new thrust::device_vector<half>(qo_len * num_qo_heads * head_dim);
    tmp = new thrust::device_vector<half>(16 * 1024 * 1024);
  }

  ~single_prefill_input_data() {
    delete tmp;
    delete O;
    delete mask;
    delete V;
    delete K;
    delete Q;
  }
};

void perf_flashinfer_single_decode(cudaStream_t& stream, single_decode_input_data* input) {
  // Provide throughput information:
  cudaError_t status = flashinfer::SingleDecodeWithKVCache(
      thrust::raw_pointer_cast(input->Q->data()), thrust::raw_pointer_cast(input->K->data()),
      thrust::raw_pointer_cast(input->V->data()), thrust::raw_pointer_cast(input->O->data()),
      input->cooperative ? thrust::raw_pointer_cast(input->tmp->data()) : nullptr, input->num_qo_heads, input->num_kv_heads,
      input->seq_len, input->head_dim, QKVLayout(input->kv_layout), PosEncodingMode(input->pos_encoding_mode),
      /*maybe_sm_scale=*/std::nullopt,
      /*rope_scale=*/1.f,
      /*rope_theta=*/1e4, stream);
  if (status != cudaSuccess) {
    std::cout << "Execution error" << std::endl;
  }
}

void perf_flashinfer_single_prefill(cudaStream_t& stream, single_prefill_input_data* input) {
  auto status = flashinfer::SinglePrefillWithKVCache<half, half>(
      thrust::raw_pointer_cast(input->Q->data()), thrust::raw_pointer_cast(input->K->data()),
      thrust::raw_pointer_cast(input->V->data()), thrust::raw_pointer_cast(input->O->data()),
      input->cooperative ? thrust::raw_pointer_cast(input->tmp->data()) : nullptr,
      nullptr, input->num_qo_heads, input->num_kv_heads, input->qo_len, input->kv_len, input->head_dim,
      input->causal, QKVLayout(input->kv_layout), PosEncodingMode(input->pos_encoding_mode),
      input->allow_fp16_qk_reduction, std::nullopt, 1.f, 1e4, stream);

  if (status != cudaSuccess) {
    std::cout << "Execution error" << std::endl;
  }
}

int main() {
  const int numGPUs = 4;
  std::vector<single_decode_input_data*> decode_data;
  std::vector<single_prefill_input_data*> prefill_data;
  for(int i = 0;i < numGPUs;i ++){
    cudaSetDevice(i);
    decode_data.push_back(new single_decode_input_data());
    prefill_data.push_back(new single_prefill_input_data());
  }

  const int iter = 100;

  {
    printf("========== one gpu one stream performance ==========\n");
    Scheduler scheduler({1});
    int gpu;
    cudaStream_t stream;
    scheduler.scheduleKernel(&gpu, &stream, false, ScheduleMode::FREE_MEMORY_SCHEDULE_MODE);
    // std::cout << "scheduled on gpu " << gpu << " stream " << stream << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    for (int i = 0; i < iter; ++ i) {
      perf_flashinfer_single_prefill(stream, prefill_data[gpu]);
      for(int j = 0;j < 30;j ++){
        perf_flashinfer_single_decode(stream, decode_data[gpu]);
      }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);

  }

  {
    printf("========== one gpu multi stream performance ==========\n");
    Scheduler scheduler({1});
    int gpu;
    cudaStream_t tensor_stream;
    cudaStream_t cuda_stream;
  
    scheduler.scheduleGPU(&gpu, &cuda_stream, &tensor_stream, ScheduleMode::FREE_MEMORY_SCHEDULE_MODE);
    // std::cout << "scheduled on gpu " << gpu << " stream " << tensor_stream << " " << cuda_stream << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, tensor_stream);


    for (int i = 0; i < iter; ++ i) {
      perf_flashinfer_single_prefill(tensor_stream, prefill_data[gpu]);
      for(int j = 0;j < 30;j ++){
        perf_flashinfer_single_decode(cuda_stream, decode_data[gpu]);
      }
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, tensor_stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);


  }

  {
    printf("========== multi gpu one stream performance ==========\n");
    Scheduler scheduler({2, 3});

    int gpu;
    cudaStream_t cudaStream, tensorStream;
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; ++ i) {
      scheduler.scheduleGPU(&gpu, &cudaStream, &tensorStream, ScheduleMode::ROUND_ROBIN_SCHEDULE_MODE);
      // std::cout << "scheduled on gpu " << gpu << " stream " << cudaStream << std::endl;

      perf_flashinfer_single_prefill(cudaStream, prefill_data[gpu]);
      perf_flashinfer_single_decode(cudaStream, decode_data[gpu]);
    }

    for(int i = 0;i < 4;i ++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);
  }

  {
    printf("========== multi gpu multi stream performance ==========\n");
    Scheduler scheduler({2, 3});

    int gpu;
    cudaStream_t cudaStream, tensorStream;
    cudaSetDevice(0);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iter; ++ i) {
      scheduler.scheduleGPU(&gpu, &cudaStream, &tensorStream, ScheduleMode::ROUND_ROBIN_SCHEDULE_MODE);
      // std::cout << "scheduled on gpu " << gpu << " stream " << cudaStream << " " << tensorStream << std::endl;

      perf_flashinfer_single_prefill(tensorStream, prefill_data[gpu]);
      perf_flashinfer_single_decode(cudaStream, decode_data[gpu]);
    }

    for(int i = 0;i < 4;i ++){
      cudaSetDevice(i);
      cudaDeviceSynchronize();
    }

    cudaSetDevice(0);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);
  }
}