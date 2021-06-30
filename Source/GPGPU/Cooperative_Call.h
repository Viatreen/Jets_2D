#pragma once

template <typename F, typename... Args>
void for_each_argument_address(F f, Args&&... args) 
{
    [](...){}((f( (void*) &std::forward<Args>(args) ), 0)...);
}

template<typename KernelFunction, typename... KernelParameters>
inline void Cooperative_Launch(
    const KernelFunction&       kernel_function,
    unsigned int      			multiProcessorCount,
	unsigned int				Thread_Count,
    KernelParameters...         parameters)
{
    void* arguments_ptrs[sizeof...(KernelParameters)];
    unsigned int arg_index = 0;
    for_each_argument_address([&](void * x) {arguments_ptrs[arg_index++] = x;}, parameters...);

    cudaLaunchCooperativeKernel<KernelFunction>(&kernel_function, multiProcessorCount, Thread_Count, arguments_ptrs);
}
