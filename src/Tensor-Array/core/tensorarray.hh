#include <cstddef>
#include <array>
#include <typeinfo>

namespace tensor_array
{
	namespace value
	{
        /**
         * \brief Static tensor.
         * \brief Not use for calulate
         */
        template <typename, unsigned int ...>
        struct TensorArray;

        template <typename T, unsigned int sz0, unsigned int ... sz>
        struct TensorArray<T, sz0, sz...>
        {
            static_assert(sz0 != 0U, "A dimension must have a value.");
            static_assert(sizeof...(sz) <= 255UL, "Max dimension is 255");
            using value_type = TensorArray<T, sz...>;
            using pointer = TensorArray<T, sz...>*;
            using reference = TensorArray<T, sz...>&;
            value_type data[sz0];
            reference operator[](size_t index)
            {
                return data[index];
            }

            constexpr const reference operator[](size_t index) const
            {
                return const_cast<const reference>(data[index]);
            }

            pointer begin()
            {
                return &data[0];
            }

            pointer end()
            {
                return &data[sz0];
            }

            const pointer cbegin() const
            {
                return &data[0];
            }

            const pointer cend() const
            {
                return &data[sz0];
            }

            pointer rbegin()
            {
                return &data[sz0 - 1];
            }

            pointer rend()
            {
                return &data[-1];
            }

            const pointer crbegin() const
            {
                return &data[sz0 - 1];
            }

            const pointer crend() const
            {
                return &data[-1];
            }
        };

        template <typename T>
        struct TensorArray<T>
        {
            using value_type = T;
            using pointer = T*;
            using reference = T&;
            value_type data;

            operator reference()
            {
                return data;
            }

            constexpr operator const reference() const
            {
                return data;
            }
        };
    }
}