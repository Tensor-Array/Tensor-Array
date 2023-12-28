#include "devices.hh"
#include <cassert>
#include <cstring>
#include <mutex>
#include <cuda_runtime.h>

namespace tensor_array
{
	namespace devices
	{
		thread_local Device default_dev = DEVICE_CPU_0;

		Device& local_device()
		{
			return default_dev;
		}

		void device_memcpy(void* dst, Device dst_dev, const void* src, Device src_dev, size_t count)
		{
			int temp;
			if (dst_dev.dev_t == CPU && src_dev.dev_t == CPU)
				std::memcpy(dst, src, count);
			else if (dst_dev.dev_t == CUDA && src_dev.dev_t == CUDA)
				assert(cudaMemcpyPeer(dst, dst_dev.index, src, src_dev.index, count) == cudaSuccess);
			else if (dst_dev.dev_t == CPU && src_dev.dev_t == CUDA)
			{
				cudaError cudaStatus = cudaGetDevice(&temp);
				cudaStatus = cudaSetDevice(src_dev.index);
				cudaStatus = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
				cudaStatus = cudaSetDevice(temp);
			}
			else if (dst_dev.dev_t == CUDA && src_dev.dev_t == CPU)
			{
				cudaError cudaStatus = cudaGetDevice(&temp);
				cudaStatus = cudaSetDevice(dst_dev.index);
				cudaStatus = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
				cudaStatus = cudaSetDevice(temp);
			}
			else
			{
				void* temp_data = std::malloc(count);
				device_memcpy(temp_data, DEVICE_CPU_0, src, src_dev, count);
				device_memcpy(dst, dst_dev, temp_data, DEVICE_CPU_0, count);
				std::free(temp_data);
			}
		}

		void device_memcpy(void* dst, Device dst_dev, const void* src, Device src_dev, size_t count, void* stream)
		{
			int temp;
			if (dst_dev.dev_t == CPU && src_dev.dev_t == CPU)
				std::memcpy(dst, src, count);
			else if (dst_dev.dev_t == CUDA && src_dev.dev_t == CUDA)
				assert(cudaMemcpyPeerAsync(dst, dst_dev.index, src, src_dev.index, count, static_cast<cudaStream_t>(stream)) == cudaSuccess);
			else if (dst_dev.dev_t == CPU && src_dev.dev_t == CUDA)
			{
				cudaError cudaStatus = cudaGetDevice(&temp);
				cudaStatus = cudaSetDevice(src_dev.index);
				cudaStatus = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, static_cast<cudaStream_t>(stream));
				cudaStatus = cudaSetDevice(temp);
			}
			else if (dst_dev.dev_t == CUDA && src_dev.dev_t == CPU)
			{
				cudaError cudaStatus = cudaGetDevice(&temp);
				cudaStatus = cudaSetDevice(dst_dev.index);
				cudaStatus = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, static_cast<cudaStream_t>(stream));
				cudaStatus = cudaSetDevice(temp);
			}
			else
			{
				void* temp_data = std::malloc(count);
				device_memcpy(temp_data, DEVICE_CPU_0, src, src_dev, count, stream);
				device_memcpy(dst, dst_dev, temp_data, DEVICE_CPU_0, count, stream);
				std::free(temp_data);
			}
		}

		void device_CUDA_get_info()
		{
			std::printf("");
			int temp;
			cudaError cudaStatus = cudaGetDevice(&temp);
			cudaDeviceProp prop;
			cudaGetDeviceProperties_v2(&prop, 0);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n",
				prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
		}
	}
}

void* operator new(size_t count, tensor_array::devices::Device dev)
{
	int temp;
	void* m_alloc_dat;
	switch (dev.dev_t)
	{
	case tensor_array::devices::CPU:
		m_alloc_dat = std::malloc(count);
		break;
	case tensor_array::devices::CUDA:
	{
		cudaError_t cuda_status = cudaGetDevice(&temp);
		cuda_status = cudaSetDevice(dev.index);
		cuda_status = cudaMalloc(&m_alloc_dat, count);
		cuda_status = cudaSetDevice(temp);
	}
	break;
	default:
		throw 0;
		break;
	}
	return m_alloc_dat;
}

void* operator new(size_t count, tensor_array::devices::Device dev, void* stream)
{
	int temp;
	void* m_alloc_dat;
	switch (dev.dev_t)
	{
	case tensor_array::devices::CPU:
		m_alloc_dat = std::malloc(count);
		break;
	case tensor_array::devices::CUDA:
	{
		cudaError_t cuda_status = cudaGetDevice(&temp);
		cuda_status = cudaSetDevice(dev.index);
		cuda_status = cudaMallocAsync(&m_alloc_dat, count, static_cast<cudaStream_t>(stream));
		cuda_status = cudaSetDevice(temp);
	}
	break;
	default:
		throw 0;
		break;
	}
	return m_alloc_dat;
}

void operator delete(void* data, tensor_array::devices::Device dev)
{
	int temp;
	switch (dev.dev_t)
	{
	case tensor_array::devices::CPU:
		std::free(data);
		break;
	case tensor_array::devices::CUDA:
	{
		cudaGetDevice(&temp);
		cudaSetDevice(dev.index);
		cudaFree(data);
		cudaSetDevice(temp);
	}
	break;
	default:
		throw 0;
		break;
	}
}

void operator delete(void* data, tensor_array::devices::Device dev, void* stream)
{
	int temp;
	switch (dev.dev_t)
	{
	case tensor_array::devices::CPU:
		std::free(data);
		break;
	case tensor_array::devices::CUDA:
	{
		cudaGetDevice(&temp);
		cudaSetDevice(dev.index);
		cudaFreeAsync(data, static_cast<cudaStream_t>(stream));
		cudaSetDevice(temp);
	}
	break;
	default:
		throw 0;
		break;
	}
}

