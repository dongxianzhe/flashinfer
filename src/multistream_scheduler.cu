#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include<nvml.h>
#include<cuda_runtime.h>

__global__ void busy_wait_kernel(long long num_clocks){
    long long start = clock64();

    volatile int x = 0;
    while((clock64() - start) < num_clocks){
        for(int i = 0;i < 100000000;i ++){
            x += 1;
        }
    };
}


long long ms2numclock(float time_ms){
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    auto clockRatekHz = deviceProp.clockRate;
    return time_ms * clockRatekHz;
}


class Scheduler {
public:
    Scheduler(int streamsPerGpu = 2);
    ~Scheduler();
    void scheduleKernel(bool isTensorCore, int& gpu, cudaStream_t& stream, std::string scheduleMode);
    void printMemInfo();

    friend std::ostream& operator<<(std::ostream& os, const Scheduler& scheduler);
private:
    int numGpus;
    int streamsPerGpu;
    int highPriority;
    int lowPriority;
    std::vector<std::vector<cudaStream_t>> streams;
    void initStreams();
    int scheduleGPUBasedFreeMemory();
    int scheduleGPUBasedGPUUsage();
    void cleanup();
};

Scheduler::Scheduler(int streamsPerGpu) : streamsPerGpu(streamsPerGpu) {
    cudaError_t cudaStatus = cudaGetDeviceCount(&numGpus);
    if (cudaStatus != cudaSuccess || numGpus <= 0) {
        throw std::runtime_error("No CUDA-enabled devices detected or failed to get device count.");
    }
    cudaStatus = cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to get stream priority range.");
    }
    streams.resize(numGpus, std::vector<cudaStream_t>(streamsPerGpu));
    initStreams();
}
Scheduler::~Scheduler() {
    cleanup();
}
void Scheduler::initStreams() {
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        // Create high priority stream for Tensor Core kernels
        cudaStreamCreateWithPriority(&streams[i][0], cudaStreamDefault, highPriority);
        // Create low priority stream for CUDA Core kernels
        cudaStreamCreateWithPriority(&streams[i][1], cudaStreamDefault, lowPriority);
    }
}
// Select the best GPU based on current load
int Scheduler::scheduleGPUBasedFreeMemory() {
    int bestGPU = 0;
    float minUsage = std::numeric_limits<float>::max();
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        cudaMemGetInfo(&freeMem, &totalMem);
        // Calculate memory usage percentage
        float usage = (totalMem - freeMem) / (float)totalMem * 100.0f;
        if (usage < minUsage) {
            minUsage = usage;
            bestGPU = i;
        }
    }
    return bestGPU;
}
int Scheduler::scheduleGPUBasedGPUUsage(){
    nvmlReturn_t result;
    result = nvmlInit();

    int bestGPU = 0;
    int minUsage = 100;
    for(int i = 0;i < numGpus;i ++){
        nvmlDevice_t device;
        result = nvmlDeviceGetHandleByIndex(i, &device);
        if(NVML_SUCCESS != result){
            printf("Scheduler::scheduleGPUBasedGPUUsage: failed to get handle for device %d\n", i);
            continue;
        }
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(device, &utilization);
        if(NVML_SUCCESS != result){
            printf("Scheduler::scheduleGPUBasedGPUUsage: failed to get handle for device %d\n", i);
            continue;
        }
        if(utilization.gpu < minUsage){
            minUsage = utilization.gpu ;
            bestGPU = i;
        }
    }
    nvmlShutdown();
    return bestGPU;
}
// Schedule a kernel on the best GPU and stream
void Scheduler::scheduleKernel(bool isTensorCore, int& gpu, cudaStream_t& stream, std::string scheduleMode) {
    if(scheduleMode == "freeMemory"){
        gpu = scheduleGPUBasedFreeMemory();
    }else if(scheduleMode == "GPUUsage"){
        gpu = scheduleGPUBasedGPUUsage();
    }else{
        gpu = scheduleGPUBasedGPUUsage();
    }
    cudaSetDevice(gpu);
    int streamIndex = isTensorCore ? 0 : 1;
    stream = streams[gpu][streamIndex];
    std::cout << "sheduled on stream " << gpu << " " << streamIndex << std::endl;
}

void Scheduler::cleanup() {
    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        for (int j = 0; j < streamsPerGpu; j++) {
            cudaStreamDestroy(streams[i][j]);
        }
    }
}

std::ostream& operator<<(std::ostream& os, const Scheduler& scheduler) {
    os << "Scheduler Info:\n";
    os << "Number of GPUs: " << scheduler.numGpus << "\n";
    os << "Streams per GPU: " << scheduler.streamsPerGpu << "\n";
    for (int i = 0; i < scheduler.numGpus; ++i) {
        os << "GPU " << i << ":\n";
        for (int j = 0; j < scheduler.streamsPerGpu; ++j) {
            os << " Stream " << j << ": " << scheduler.streams[i][j] << "\n"; 
        } 
    }
    return os; 
}

void Scheduler::printMemInfo(){
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    std::vector<cudaDeviceProp> deviceProps(deviceCount);
    std::vector<size_t>freeMemory(deviceCount);
    std::vector<size_t>totalMemory(deviceCount);

    for(int i = 0;i < deviceCount;i ++){
        cudaSetDevice(i);

        cudaMemGetInfo(&freeMemory[i], &totalMemory[i]);

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        deviceProps[i] = prop;
    }

    std::cout << "------------------------------" << std::endl;
    for(int i = 0;i < deviceCount;i ++){
        const cudaDeviceProp& prop = deviceProps[i];
        std::cout << "device " << i << "(" << prop.name << ")" << "memory info: " << std::endl;
        std::cout << "total memory: " << totalMemory[i] / 1024 / 1024 << " MB" << std::endl;
        std::cout << "free  memory: " << freeMemory[i]  / 1024 / 1024 << " MB" << std::endl;
        std::cout << "used  memory: " << (totalMemory[i] - freeMemory[i]) / 1024 / 1024 << " MB" << std::endl;
    }
    std::cout << "------------------------------" << std::endl;
}

int main(){
    Scheduler scheduler;
    std::cout << scheduler << std::endl;

    int num_kernels = 10;
    for(int i = 0;i < num_kernels;i ++){
        int gpu_id;
        cudaStream_t stream;
        // scheduler.printMemInfo();
        scheduler.scheduleKernel(false, gpu_id, stream, "GPUUsage");

        cudaSetDevice(gpu_id);
        busy_wait_kernel<<<1024, 128, 0, stream>>>(ms2numclock(5000));
        sleep(1);
    }
    for(int i = 0;i < 4;i ++){
        cudaSetDevice(i);
        cudaDeviceSynchronize();
    }
}