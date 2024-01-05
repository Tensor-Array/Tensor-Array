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

#include "sequential.hh"

namespace tensor_array
{
	namespace layers
	{
        SequentialImpl::SequentialImpl(const std::initializer_list<LayerInSequential>& function_list)
        {
            for (auto& it : function_list)
            {
                auto index = this->function_list.size();
                this->function_list.emplace_back(it);
                this->map_layer.insert(std::make_pair("layer" + std::to_string(index), it.get_shared()));
            }
        }

        void SequentialImpl::insert(const LayerInSequential& layer)
        {
            auto index = this->function_list.size();
            this->map_layer.insert(std::make_pair("layer" + std::to_string(index), layer.get_shared()));
            this->function_list.push_back(layer);
        }

        void SequentialImpl::insert(LayerInSequential&& layer_ptr)
        {
            auto index = this->function_list.size();
            this->map_layer.insert(std::make_pair("layer" + std::to_string(index), layer_ptr.get_shared()));
            this->function_list.push_back(std::forward<LayerInSequential>(layer_ptr));
        }

        LayerInSequential& SequentialImpl::get(std::size_t index)
        {
            return this->function_list[index];
        }

        value::Tensor SequentialImpl::calculate(const value::Tensor& input_tensor)
        {
            value::Tensor temp = input_tensor;
            for (auto& item : this->function_list)
                temp = item(input_tensor);
            return temp;
        }

        Sequential::Sequential(std::initializer_list<LayerInSequential>&& list_holder):
            LayerHolder(std::forward<std::initializer_list<LayerInSequential>>(list_holder))
        {
        }
	}
}