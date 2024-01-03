#include "extern_type_map.hh"

namespace tensor_array
{
    namespace value
    {
        std::unordered_map<std::type_index, std::size_t> dynamic_type_size
        {
            {typeid(bool), sizeof(bool)},
            {typeid(int), sizeof(int)},
            {typeid(unsigned int), sizeof(unsigned int)},
        };
    }
}