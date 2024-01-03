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
