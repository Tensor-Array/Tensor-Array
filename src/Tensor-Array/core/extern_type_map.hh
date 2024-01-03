#include <unordered_map>
#include <typeindex>

namespace tensor_array
{
    namespace value
    {
        extern std::unordered_map<std::type_index, std::size_t> dynamic_type_size;
    }
}
