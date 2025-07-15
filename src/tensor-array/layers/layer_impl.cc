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

#include <filesystem>
#include <iostream>
#include "layer_impl.hh"

namespace tensor_array
{
    namespace layers
    {
        void LayerImpl::layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape)
        {
        }
        
        void LayerImpl::update_weight(float eta)
        {
            for (auto& it : this->map_layer)
                it.second->update_weight(eta);
            for (auto& it : this->map_tensor)
            {
                if (!it.second->has_tensor()) continue;
                const tensor_array::value::TensorBase& temp = it.second->get_buffer();
                value::Tensor temp_grad = it.second->get_grad();
                if (temp_grad.has_tensor())
                {
                    *it.second -= multiply(value::values(temp.shape(), eta).tensor_cast(temp.type()), temp_grad);
                    *it.second = it.second->clone();
                }
            }
        }

        void LayerImpl::load_data(const std::string& str)
        {
            for (auto& it : this->map_layer)
                it.second->load_data(str + '/' + it.first);
            for (auto& it : this->map_tensor)
            {
                std::string next_dir = (str + '/' + it.first);
                value::Tensor test = tensor_array::value::tensor_file_load(next_dir.c_str());
                if (test.has_tensor())
                    *it.second = test;
            }
            this->is_running = false;
        }

        void LayerImpl::save_data(const std::string& str) const
        {
            for (auto& it : this->map_layer)
                it.second->save_data(str + '/' + it.first);
            for (auto& it : this->map_tensor)
            {
                if (!std::filesystem::exists(str))
                {
                    std::filesystem::create_directories(str);
                }
                it.second->save((str + '/' + it.first).c_str());
            }
        }

}
}
