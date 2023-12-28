#pragma once

namespace tensor_array
{
	namespace datatype
	{
		enum DataType : unsigned char
		{
			s_bool = 0,
			s_char = 1,
			s_short = 2,
			s_int = 3,
			s_long = 4,
			s_float = 5,
			s_double = 6,
			s_long_long = 8,
			u_bool = 128,
			u_char = 129,
			u_short = 130,
			u_int = 131,
			u_long = 132,
			u_long_long = 136,
		};

		const std::type_info& warp_type(const DataType&);
		DataType warp_type(const std::type_info&);
	}
}