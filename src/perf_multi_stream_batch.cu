#include <iostream>
#include <thrust/device_vector.h>

#include <cupti.h>

#include"multistream_scheduler.cuh"
#include "flashinfer_ops.cuh"

using flashinfer::PosEncodingMode;
using flashinfer::QKVLayout;

constexpr auto pos_encoding_mode = flashinfer::PosEncodingMode::kNone;
constexpr QKVLayout kv_layout = flashinfer::QKVLayout::kNHD;

struct batch_decode_input_data {
  size_t head_dim = 128;

  size_t seqlen = 1024;
  size_t batch_size = 4;
  size_t page_size = 64;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  bool cooperative = false;
  size_t pages_per_seq = (seqlen + page_size - 1) / page_size;
  size_t num_pages = pages_per_seq * batch_size;

  std::vector<int32_t>* kv_indptr_host = nullptr;
  std::vector<int32_t>* kv_indicies_host = nullptr;
  std::vector<int32_t>* kv_last_page_len_host = nullptr;

  thrust::device_vector<half>* kv_data = nullptr;
  thrust::device_vector<int32_t>* kv_indptr = nullptr;
  thrust::device_vector<int32_t>* kv_indices = nullptr;
  thrust::device_vector<int32_t>* kv_last_page_len = nullptr;

  flashinfer::paged_kv_t<flashinfer::PageStorage::kIndices, half, int32_t>* paged_kv = nullptr;

  thrust::device_vector<half>* q = nullptr;
  thrust::device_vector<half>* o = nullptr;

  size_t float_workspace_size_in_bytes = 32 * 1024 * 1024;
  thrust::device_vector<char>* float_buffer = nullptr;
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;
  thrust::device_vector<char>* int_buffer = nullptr;

  // Allocate input data:
  batch_decode_input_data() {
    kv_indptr_host = new std::vector<int32_t>();
    kv_indicies_host      = new std::vector<int32_t>();
    kv_last_page_len_host = new std::vector<int32_t>();
    kv_indptr_host->push_back(0);
    for (size_t i = 0; i < batch_size; ++i) {
        for (size_t p = 0; p < pages_per_seq; ++p) {
            kv_indicies_host->push_back(i * pages_per_seq + p);
        }
        kv_indptr_host->push_back(kv_indptr_host->back() + pages_per_seq);
        kv_last_page_len_host->push_back((seqlen - 1) % page_size + 1);
    }

    kv_data          = new thrust::device_vector<half>   (num_pages * 2 * num_kv_heads * page_size * head_dim);
    kv_indptr        = new thrust::device_vector<int32_t>(*kv_indptr_host);
    kv_indices       = new thrust::device_vector<int32_t>(*kv_indicies_host);
    kv_last_page_len = new thrust::device_vector<int32_t>(*kv_last_page_len_host);
    paged_kv = new flashinfer::paged_kv_t<flashinfer::PageStorage::kIndices, half, int32_t>(
        num_kv_heads, page_size, head_dim, batch_size, kv_layout,
        thrust::raw_pointer_cast(kv_data->data()), thrust::raw_pointer_cast(kv_indices->data()),
        thrust::raw_pointer_cast(kv_indptr->data()),
        thrust::raw_pointer_cast(kv_last_page_len->data()));
    
     q = new thrust::device_vector<half>(batch_size * num_qo_heads * head_dim);
     o = new thrust::device_vector<half>(batch_size * num_qo_heads * head_dim);

     float_buffer = new thrust::device_vector<char>(float_workspace_size_in_bytes);
     int_buffer   = new thrust::device_vector<char>(int_workspace_size_in_bytes);
  }

  ~batch_decode_input_data() {
    delete kv_indptr_host;
    delete kv_indicies_host;
    delete kv_last_page_len_host;
    delete kv_data ;
    delete kv_indptr;
    delete kv_indices;
    delete kv_last_page_len;
    delete q;
    delete o;
    delete float_buffer;
    delete int_buffer;
  }
};

struct batch_prefill_input_data {
  size_t kv_len = 2048;
  size_t qo_len = kv_len;
  size_t batch_size = 4;
  size_t num_qo_heads = 32;
  size_t num_kv_heads = 32;
  size_t head_dim = 128;
  size_t pos_encoding_mode = 0;
  size_t kv_layout = 0;
  bool causal = true;
  bool cooperative = false;
  bool allow_fp16_qk_reduction = false;
  size_t float_workspace_size_in_bytes = 128 * 1024 * 1024;
  size_t int_workspace_size_in_bytes = 8 * 1024 * 1024;

  thrust::device_vector<half>* Q = nullptr;
  thrust::device_vector<half>* K = nullptr;
  thrust::device_vector<half>* V = nullptr;
  thrust::device_vector<half>* O = nullptr;
  thrust::device_vector<uint8_t>* float_workspace = nullptr;
  thrust::device_vector<uint8_t>* int_workspace = nullptr;
  std::vector<int32_t>* qo_indptr_h = nullptr;
  std::vector<int32_t>* kv_indptr_h = nullptr;
  thrust::device_vector<int32_t>* qo_indptr_d = nullptr;
  thrust::device_vector<int32_t>* kv_indptr_d = nullptr;

  batch_prefill_input_data() {
    Q = new thrust::device_vector<half> (batch_size * qo_len * num_qo_heads * head_dim);
    K = new thrust::device_vector<half> (batch_size * kv_len * num_kv_heads * head_dim);
    V = new thrust::device_vector<half> (batch_size * kv_len * num_kv_heads * head_dim);
    O = new thrust::device_vector<half> (batch_size * qo_len * num_qo_heads * head_dim);
    float_workspace = new thrust::device_vector<uint8_t> (float_workspace_size_in_bytes);
    int_workspace   = new thrust::device_vector<uint8_t> (int_workspace_size_in_bytes);
    qo_indptr_h = new std::vector<int32_t>(batch_size + 1);
    kv_indptr_h = new std::vector<int32_t>(batch_size + 1);

    for (uint32_t i = 0; i <= batch_size; ++i) {
        (*qo_indptr_h)[i] = i * qo_len;
        (*kv_indptr_h)[i] = i * kv_len;
    }

    qo_indptr_d = new thrust::device_vector<int32_t>(*qo_indptr_h);
    kv_indptr_d = new thrust::device_vector<int32_t>(*kv_indptr_h);
  }

  ~batch_prefill_input_data() {
    delete Q;
    delete K;
    delete V;
    delete O;
    delete float_workspace;
    delete int_workspace  ;
    delete qo_indptr_d;
    delete kv_indptr_d;
  }
};

void perf_flashinfer_batch_decode(cudaStream_t& stream, batch_decode_input_data* input) {
  if (input->cooperative) {
    flashinfer::BatchDecodeHandler handler;
    flashinfer::BatchDecodeHandlerBeginForward<flashinfer::PageStorage::kIndices, half, half, half, int32_t>(
        &handler, (void*)thrust::raw_pointer_cast(input->float_buffer->data()),
        input->float_workspace_size_in_bytes, (void*)thrust::raw_pointer_cast(input->int_buffer->data()),
        input->int_workspace_size_in_bytes, input->kv_indptr_host->data(), input->kv_last_page_len_host->data(),
        input->batch_size, input->num_qo_heads, input->num_kv_heads, input->head_dim, input->page_size, pos_encoding_mode);

    cudaError_t status = flashinfer::BatchDecodeWithPagedKVCacheWrapper<flashinfer::PageStorage::kIndices, half, half, half, int32_t>(
              &handler, thrust::raw_pointer_cast(input->q->data()), /*q_offset=*/nullptr, *(input->paged_kv),
              thrust::raw_pointer_cast(input->o->data()), /*lse=*/nullptr, input->num_qo_heads, pos_encoding_mode);
    if (status != cudaSuccess) {
        std::cout << "Execution error" << std::endl;
    }
  }
  else{
    cudaError_t status = flashinfer::BatchDecodeWithPagedKVCacheNoSplitKV<flashinfer::PageStorage::kIndices, half, half, half, int32_t>(
            thrust::raw_pointer_cast(input->q->data()), /*q_offset=*/nullptr, *(input->paged_kv),
            flashinfer::kv_partition_info_t<int32_t>(), thrust::raw_pointer_cast(input->o->data()),
            /*lse=*/nullptr, input->num_qo_heads, pos_encoding_mode);
    if (status != cudaSuccess) {
        std::cout << "Execution error" << std::endl;
    }

  }
}

void perf_flashinfer_batch_prefill_with_ragged_kv(cudaStream_t& stream, batch_prefill_input_data* input) {
    flashinfer::BatchPrefillHandler handler;
    handler.BeginForward<half>(
        thrust::raw_pointer_cast(input->float_workspace->data()), input->float_workspace_size_in_bytes,
        thrust::raw_pointer_cast(input->int_workspace->data()), input->int_workspace_size_in_bytes,
        input->qo_indptr_h->data(), input->kv_indptr_h->data(), input->batch_size, input->num_qo_heads, input->num_kv_heads, input->head_dim,
        1);

    auto status = flashinfer::BatchPrefillWithRaggedKVCacheWrapper<half, half, half, int32_t>(
        &handler, thrust::raw_pointer_cast(input->Q->data()), thrust::raw_pointer_cast(input->qo_indptr_d->data()),
        thrust::raw_pointer_cast(input->K->data()), thrust::raw_pointer_cast(input->V->data()),
        thrust::raw_pointer_cast(input->kv_indptr_d->data()),
        /*q_offset=*/nullptr, /*k_rope_pos_offset=*/nullptr, thrust::raw_pointer_cast(input->O->data()),
        /*lse=*/nullptr, input->batch_size, input->num_qo_heads, input->num_kv_heads, input->head_dim, input->causal,
        QKVLayout(input->kv_layout), PosEncodingMode(input->pos_encoding_mode), input->allow_fp16_qk_reduction);
    if (status != cudaSuccess) {
        std::cout << "Execution error" << std::endl;
    }

}

int main() {
  Scheduler scheduler({1});
  int gpu;
  cudaStream_t tensor_stream;
  cudaStream_t cuda_stream;

  scheduler.scheduleGPU(&gpu, &cuda_stream, &tensor_stream, ScheduleMode::FREE_MEMORY_SCHEDULE_MODE);

  const int iter = 100;
  cudaSetDevice(gpu);
  batch_prefill_input_data prefill_data;
  batch_decode_input_data decode_data;

  {
    printf("========== one gpu one stream performance ==========\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, tensor_stream);
    for (int i = 0; i < iter; ++ i) {
      perf_flashinfer_batch_prefill_with_ragged_kv(tensor_stream, &prefill_data);
      perf_flashinfer_batch_decode(tensor_stream, &decode_data);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, tensor_stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);

  }

  {
    printf("========== one gpu multi stream performance ==========\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);cudaEventCreate(&stop);
    cudaEventRecord(start, tensor_stream);

    for (int i = 0; i < iter; ++ i) {
      perf_flashinfer_batch_prefill_with_ragged_kv(tensor_stream, &prefill_data);
      perf_flashinfer_batch_decode(cuda_stream, &decode_data);
    }
    cudaDeviceSynchronize();

    cudaEventRecord(stop, tensor_stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("elapsed time %f ms\n", milliseconds);
  }

}