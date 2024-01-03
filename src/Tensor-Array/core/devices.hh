#pragma once

#ifdef __WIN32__
#ifdef CUDA_ML_EXPORTS
#define CUDA_ML_API __declspec(dllexport)
#else
#define CUDA_ML_API __declspec(dllimport)
#endif
#else
#define CUDA_ML_API
#endif

namespace tensor_array
{
	namespace devices
	{
		enum DeviceType
		{
			CPU,
			CUDA
		};

		struct Device
		{
			DeviceType dev_t;
			int index;
		};

		constexpr Device DEVICE_CPU_0{ CPU,0 };

		CUDA_ML_API Device& local_device();

		void device_memcpy(void*, Device, const void*, Device, size_t);

		void device_memcpy(void*, Device, const void*, Device, size_t, void*);

		void device_memset(void*, Device, int, size_t);

		void device_memset(void*, Device, int, size_t, void*);

		CUDA_ML_API void device_CUDA_get_info();
	}
}

void* operator new(size_t, tensor_array::devices::Device);

void* operator new(size_t, tensor_array::devices::Device, void*);

void operator delete(void*, tensor_array::devices::Device);

void operator delete(void*, tensor_array::devices::Device, void*);

#undef CUDA_ML_API