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

#include <initializer_list>

namespace tensor_array
{
    namespace wrapper
    {
        template<class _E>
    class initializer_wrapper
    {
    public:
        typedef _E 		value_type;
        typedef const _E& 	reference;
        typedef const _E& 	const_reference;
        typedef size_t 		size_type;
        typedef const _E* 	iterator;
        typedef const _E* 	const_iterator;

    private:
#ifdef __GNUC__
        iterator			_M_array;
        size_type			_M_len;
#endif

    public:
        constexpr initializer_wrapper(const_iterator __a, size_type __l)
        : _M_array(__a), _M_len(__l) { }

        constexpr initializer_wrapper(const_iterator __begin, const_iterator __end)
        : _M_array(__begin), _M_len(__end - __begin) { }
        
        constexpr initializer_wrapper() noexcept: _M_array(0), _M_len(0) { }
        
        // Number of elements.
        constexpr size_type
        size() const noexcept { return _M_len; }
        
        // First element.
        constexpr const_iterator
        begin() const noexcept { return _M_array; }
        
        // One past the last element.
        constexpr const_iterator
        end() const noexcept { return begin() + size(); }

        constexpr operator std::initializer_list<_E>() const { return reinterpret_cast<const std::initializer_list<_E>&>(*this); }
    };
    }
}