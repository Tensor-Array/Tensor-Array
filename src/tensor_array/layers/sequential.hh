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

#include "layer_holder.hh"
#pragma once

namespace tensor_array
{
	namespace layers
	{
        using LayerInSequential = LayerHolder<TensorCalculateLayerImpl>;

        class CUDA_ML_API SequentialImpl final :
            public TensorCalculateLayerImpl
        {
        private:
            std::vector<LayerInSequential> function_list;
        public:
            SequentialImpl() = default;
            SequentialImpl(const std::initializer_list<LayerInSequential>&);
            void insert(const LayerInSequential&);
            void insert(LayerInSequential&&);
            LayerInSequential& get(std::size_t);
            value::Tensor calculate(const value::Tensor&) override;
        };

        class CUDA_ML_API Sequential : public LayerHolder<SequentialImpl>
        {
        public:
            Sequential() = default;
            Sequential(std::initializer_list<LayerInSequential>&& list_holder);
        };
	}
}

