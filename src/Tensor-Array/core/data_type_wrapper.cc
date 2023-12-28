#include <typeinfo>
#include "data_type_wrapper.hh"

namespace tensor_array
{
	namespace datatype
	{
		const std::type_info& warp_type(const DataType& dtype)
		{
			switch (dtype)
			{
			case s_bool: case u_bool:
				return typeid(bool);
			case s_char:
				return typeid(char);
			case s_short:
				return typeid(short);
			case s_int:
				return typeid(int);
			case s_long:
				return typeid(long);
			case s_float:
				return typeid(float);
			case s_double:
				return typeid(double);
			case s_long_long:
				return typeid(long long);
			case u_char:
				return typeid(unsigned char);
			case u_short:
				return typeid(unsigned short);
			case u_int:
				return typeid(unsigned);
			case u_long:
				return typeid(unsigned long);
			case u_long_long:
				return typeid(unsigned long long);
			default:
				throw 0;
			}
		}

		DataType warp_type(const std::type_info& dtype)
		{
			if (dtype == typeid(bool)) return s_bool;
			if (dtype == typeid(char)) return s_char;
			if (dtype == typeid(short)) return s_short;
			if (dtype == typeid(int)) return s_int;
			if (dtype == typeid(long)) return s_long;
			if (dtype == typeid(float)) return s_float;
			if (dtype == typeid(double)) return s_double;
			if (dtype == typeid(long long)) return s_long_long;
			if (dtype == typeid(unsigned char)) return u_char;
			if (dtype == typeid(unsigned short)) return u_short;
			if (dtype == typeid(unsigned)) return u_int;
			if (dtype == typeid(unsigned long)) return u_long;
			if (dtype == typeid(unsigned long long)) return u_long_long;
			throw 0;
		}
	}
}