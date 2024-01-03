/*
Copyright 2024 TensorArray-Creators

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "data_type_wrapper.hh"
#include "extern_type_map.hh"

#define LOOP(seq) END(A seq)
#define BODY(x) ADD_CODE(x)
#define A(x) BODY(x) B
#define B(x) BODY(x) A
#define A_END
#define B_END
#define END(...) END_(__VA_ARGS__)
#define END_(...) __VA_ARGS__##_END

typedef __nv_bfloat16 bfloat16;

#define USING_DATA_TYPE_NVIDIA_FLOAT_8 (__nv_fp8_e5m2)(__nv_fp8_e4m3)
#define USING_DATA_TYPE_NVIDIA_FLOAT (half)(bfloat16)
#define USING_DATA_TYPE_FLOAT (float)(double)
#define USING_DATA_TYPE_SINT (int8_t)(int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT (uint8_t)(uint16_t)(uint32_t)(uint64_t)
#define USING_DATA_TYPE USING_DATA_TYPE_SINT USING_DATA_TYPE_UINT USING_DATA_TYPE_FLOAT USING_DATA_TYPE_NVIDIA_FLOAT

namespace tensor_array
{
	namespace datatype
	{

#define ADD_CODE(TYPE)\
const std::type_info& get_dtype_##TYPE\
()\
{\
const std::type_info& info = typeid(TYPE);\
std::type_index test = typeid(TYPE);\
if (value::dynamic_type_size.find(test) == value::dynamic_type_size.end())\
value::dynamic_type_size.insert(std::make_pair(test, sizeof(TYPE)));\
return info;\
}
		LOOP(USING_DATA_TYPE)
#undef ADD_CODE


        const std::type_info& warp_type(DataType dtype)
		{
			switch (dtype)
			{
			case BOOL_DTYPE:
				return typeid(bool);
			case S_INT_8:
				return get_dtype_int8_t();
			case S_INT_16:
				return get_dtype_int16_t();
			case S_INT_32:
				return get_dtype_int32_t();
			case FLOAT_DTYPE:
				return get_dtype_float();
			case DOUBLE_DTYPE:
				return get_dtype_double();
			case HALF_DTYPE:
				return get_dtype_half();
			case S_INT_64:
				return get_dtype_int64_t();
			case BF16_DTYPE:
				return get_dtype_bfloat16();
			case U_INT_8:
				return get_dtype_uint8_t();
			case U_INT_16:
				return get_dtype_uint16_t();
			case U_INT_32:
				return get_dtype_uint32_t();
			case U_INT_64:
				return get_dtype_uint64_t();
			default:
				throw 0;
			}
		}

		DataType warp_type(const std::type_info& dtype)
		{
			if (dtype == typeid(bool)) return BOOL_DTYPE;
			if (dtype == typeid(std::int8_t)) return S_INT_8;
			if (dtype == typeid(std::int16_t)) return S_INT_16;
			if (dtype == typeid(std::int32_t)) return S_INT_32;
			if (dtype == typeid(std::int64_t)) return S_INT_64;
			if (dtype == typeid(float)) return FLOAT_DTYPE;
			if (dtype == typeid(double)) return DOUBLE_DTYPE;
			if (dtype == typeid(std::uint8_t)) return U_INT_8;
			if (dtype == typeid(std::uint16_t)) return U_INT_16;
			if (dtype == typeid(std::uint32_t)) return U_INT_32;
			if (dtype == typeid(std::uint64_t)) return U_INT_64;
			throw 0;
		}
	}
}

#undef USING_DATA_TYPE

#undef LOOP
#undef BODY
#undef A
#undef B
#undef A_END
#undef B_END
#undef END
#undef END_
