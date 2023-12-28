#include <filesystem>
#include <iostream>
#include "layer_impl.hh"

namespace tensor_array
{
    namespace layers
    {
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
                    *it.second = it.second->new_grad_copy();
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
