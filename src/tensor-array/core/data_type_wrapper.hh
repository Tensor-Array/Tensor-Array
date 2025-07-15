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

#include <typeinfo>
#pragma once

namespace tensor_array
{
	namespace datatype
	{
		enum DataType : unsigned char
		{
			BOOL_DTYPE = 0,
			S_INT_8 = 1,
			S_INT_16 = 2,
			BF16_DTYPE = 3,
			S_INT_32 = 4,
			FLOAT_DTYPE = 5,
			DOUBLE_DTYPE = 6,
			HALF_DTYPE = 7,
			S_INT_64 = 8,
			U_INT_8 = 129,
			U_INT_16 = 130,
			U_INT_32 = 132,
			U_INT_64 = 136,
		};

		const std::type_info& warp_type(DataType);
		DataType warp_type(const std::type_info&);
	}
}
