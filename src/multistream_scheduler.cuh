#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <limits>
#include <stdexcept>
#include <unistd.h>
#include<nvml.h>
#include<cuda_runtime.h>
#include<random>


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

enum ScheduleMode{
    FREE_MEMORY_SCHEDULE_MODE = 0,
    GPU_USAGE_SCHEDULE_MODE = 1,
    RANDOM_SCHEDULE_MODE = 2,
    ROUND_ROBIN_SCHEDULE_MODE = 3
};

class Scheduler {
public:
    Scheduler();
    Scheduler(std::vector<int>gpus);
    ~Scheduler();
    void scheduleKernel(int* gpu, cudaStream_t* stream, bool isTensorCore, ScheduleMode scheduleMode);
    void scheduleGPU(int* gpu, cudaStream_t* cudaStream, cudaStream_t* tensorStream, ScheduleMode scheduleMode);
    void printMemInfo();

    friend std::ostream& operator<<(std::ostream& os, const Scheduler& scheduler);
private:
    int numGpus;
    const int streamsPerGpu = 2;
    int highPriority;
    int lowPriority;
    std::vector<std::vector<cudaStream_t>> streams;
    std::vector<int> schedule_gpu_list;
private:
    void initStream();
    int scheduleGPUBasedFreeMemory();
    int scheduleGPUBasedGPUUsage();
    int scheduleGPURandom();
    int scheduleGPURoundRobin();
    void dispatchScheduleMode(int* gpu, ScheduleMode scheduleMode);
    void cleanup();
};

void Scheduler::initStream(){
    cudaError_t cudaStatus = cudaGetDeviceCount(&numGpus);
    if (cudaStatus != cudaSuccess || numGpus <= 0) {
        throw std::runtime_error("No CUDA-enabled devices detected or failed to get device count.");
    }
    cudaStatus = cudaDeviceGetStreamPriorityRange(&lowPriority, &highPriority);
    if (cudaStatus != cudaSuccess) {
        throw std::runtime_error("Failed to get stream priority range.");
    }
    streams.resize(numGpus, std::vector<cudaStream_t>(streamsPerGpu));

    for (int i = 0; i < numGpus; i++) {
        cudaSetDevice(i);
        // Create high priority stream for Tensor Core kernels
        cudaStreamCreateWithPriority(&streams[i][0], cudaStreamDefault, highPriority);
        // Create low priority stream for CUDA Core kernels
        cudaStreamCreateWithPriority(&streams[i][1], cudaStreamDefault, lowPriority);
    }
}

Scheduler::Scheduler(){
    initStream();
    for(int i = 0;i < numGpus;i ++)schedule_gpu_list.push_back(i);
}

Scheduler::Scheduler(std::vector<int>schedule_gpu_list){
    initStream();
    this->schedule_gpu_list = schedule_gpu_list;
}

Scheduler::~Scheduler() {
    cleanup();
}

int Scheduler::scheduleGPUBasedFreeMemory() {
    // find GPU having largest free memory
    int bestGPU = 0;
    float minUsage = std::numeric_limits<float>::max();
    for(auto i : schedule_gpu_list){
        cudaSetDevice(i);
        size_t freeMem, totalMem;
        auto err = cudaMemGetInfo(&freeMem, &totalMem);
        if(err != cudaSuccess){
            printf("Scheduler::scheduleGPUBasedFreeMemory: failed to cudaMemGetInfo %d\n", i);
        }
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
    // find the least compute usage GPU
    int bestGPU = 0;
    int minUsage = 100;
    for(auto i : schedule_gpu_list){
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


int Scheduler::scheduleGPURandom(){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, schedule_gpu_list.size() - 1); // includes chedule_gpu_list.size() - 1
    int i = distrib(gen);
    return schedule_gpu_list[i];
};

int Scheduler::scheduleGPURoundRobin(){
    static int nextGPU;
    int gpu = schedule_gpu_list[nextGPU];
    nextGPU = (nextGPU + 1) % schedule_gpu_list.size();
    return gpu;
};

void Scheduler::dispatchScheduleMode(int* gpu, ScheduleMode scheduleMode) {
    switch (scheduleMode)
    {
    case FREE_MEMORY_SCHEDULE_MODE:
        *gpu = scheduleGPUBasedFreeMemory();
        break;
    case GPU_USAGE_SCHEDULE_MODE:
        *gpu = scheduleGPUBasedGPUUsage();
        break;
    case RANDOM_SCHEDULE_MODE:
        *gpu = scheduleGPURandom();
        break;
    case ROUND_ROBIN_SCHEDULE_MODE:
        *gpu = scheduleGPURoundRobin();
        break;
    default:
        fprintf(stderr, "warnning invalid schedule mode\n");
        *gpu = scheduleGPUBasedFreeMemory();
        break;
    }
    cudaSetDevice(*gpu);
}

void Scheduler::scheduleGPU(int* gpu, cudaStream_t* cudaStream, cudaStream_t* tensorStream, ScheduleMode scheduleMode){
    dispatchScheduleMode(gpu, scheduleMode);
    *cudaStream   = streams[*gpu][1];
    *tensorStream = streams[*gpu][0];
}

// Schedule a kernel on the best GPU and stream
void Scheduler::scheduleKernel(int* gpu, cudaStream_t* stream, bool isTensorCore, ScheduleMode scheduleMode) {
    dispatchScheduleMode(gpu, scheduleMode);
    int streamIndex = isTensorCore ? 0 : 1;
    *stream = streams[*gpu][streamIndex];
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
    for (auto i : scheduler.schedule_gpu_list) {
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


// int main(){
//     Scheduler scheduler;
//     std::cout << scheduler << std::endl;

//     int num_kernels = 10;
//     for(int i = 0;i < num_kernels;i ++){
//         int gpu_id;
//         cudaStream_t stream;
//         // scheduler.printMemInfo();
//         // scheduler.scheduleKernel(false, gpu_id, stream, "GPUUsage");
//         scheduler.scheduleKernel(gpu_id, stream, false, "freeMemory");

//         cudaSetDevice(gpu_id);
//         busy_wait_kernel<<<1024, 128, 0, stream>>>(ms2numclock(5000));

//         sleep(1);
//     }
//     for(int i = 0;i < 4;i ++){
//         cudaSetDevice(i);
//         cudaDeviceSynchronize();
//     }
// }