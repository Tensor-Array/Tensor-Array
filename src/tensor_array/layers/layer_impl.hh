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

#include <string>
#include <unordered_map>
#include <tensor_array/core/tensor.hh>
#include <utility>
#pragma once

#ifdef __WIN32__
#ifdef CUDA_ML_EXPORTS
#define CUDA_ML_API __declspec(dllexport)
#else
#define CUDA_ML_API __declspec(dllimport)
#endif
#else
#define CUDA_ML_API
#endif

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API LayerImpl
        {
        private:
            bool is_running = false;
            template <class T>
            friend class LayerHolder;
        protected:
            std::unordered_map<std::string, value::Tensor*> map_tensor;
            std::unordered_map<std::string, std::shared_ptr<LayerImpl>> map_layer;
        public:
            virtual ~LayerImpl() = default;
            virtual void update_weight(float = 0.001f) final;
            virtual void load_data(const std::string&) final;
            virtual void save_data(const std::string&) const final;
            inline LayerImpl() = default;
            virtual void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&&);
            constexpr LayerImpl(const LayerImpl&) = delete;
            constexpr LayerImpl(LayerImpl&&) = delete;
            inline LayerImpl operator=(const LayerImpl&) = delete;
            inline LayerImpl operator=(LayerImpl&&) = delete;
        };

        struct CalculateStruct
        {
            virtual ~CalculateStruct() = default;
            virtual value::Tensor calculate(const value::Tensor&) = 0;
        };

        struct TensorCalculateLayerImpl :
            public LayerImpl,
            public CalculateStruct
        {};
    }
}