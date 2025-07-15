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

#include "linear.hh"
#include "layer_utility.hh"
#pragma once

namespace tensor_array
{
    namespace layers
    {
        class CUDA_ML_API ConvolutionLayerImpl :
            public TensorCalculateLayerImpl
        {
        protected:
            const unsigned char dim;
            const value::dimension kernel_size;
            const unsigned int filter;
            const value::dimension strides;
            const value::dimension padding;
            const value::dimension dilation;
            value::Tensor kernel;
            value::Tensor bias;
            ConvolutionLayerImpl(unsigned char dim, const value::dimension&, unsigned int, const value::dimension & = value::dimension(), const value::dimension& = {0,0,0}, const value::dimension & = value::dimension());
        public:
            value::Tensor calculate(const value::Tensor&) override final;
        };

        class CUDA_ML_API Conv1D_Impl final :
            public ConvolutionLayerImpl
        {
        public:
            Conv1D_Impl(const value::dimension&, unsigned int, const value::dimension & = value::dimension(), const value::dimension& = value::dimension());
            void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape) override;
        };

        using Conv1D = LayerHolder<Conv1D_Impl>;

        class CUDA_ML_API Conv2D_Impl final :
            public ConvolutionLayerImpl
        {
        public:
            Conv2D_Impl(const value::dimension&, unsigned int, const value::dimension & = value::dimension(), const value::dimension& = value::dimension());
            void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape) override;
        };

        using Conv2D = LayerHolder<Conv2D_Impl>;

        class CUDA_ML_API Conv3D_Impl final :
            public ConvolutionLayerImpl
        {
        public:
            Conv3D_Impl(const value::dimension&, unsigned int, const value::dimension & = value::dimension(), const value::dimension& = value::dimension());
            void layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& vector_shape) override;
        };

        using Conv3D = LayerHolder<Conv3D_Impl>;
    }
}

