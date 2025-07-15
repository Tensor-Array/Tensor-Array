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

#include "normalization.hh"

namespace tensor_array
{
	namespace layers
	{
        void NormalizationImpl::layer_init(std::vector<std::pair<std::initializer_list<unsigned int>, const std::type_info&>>&& shape_input)
        {
            std::vector<unsigned int> temp = shape_input[0].first;
            const std::type_info& type = shape_input[0].second;
            for (auto& it : dims_mean)
                temp[it] = 1U;
            if (!this->gamma.has_tensor())
                this->gamma = value::values(temp, 1).tensor_cast(type);
            if (!this->beta.has_tensor())
                this->beta = value::values(temp, 0).tensor_cast(type);
            if (!this->moving_mean.has_tensor())
                this->moving_mean = value::values(temp, 0).tensor_cast(type);
            if (!this->moving_variance.has_tensor())
                this->moving_variance = value::values(temp, 1).tensor_cast(type);
            this->moving_mean = this->moving_mean.clone();
            this->moving_variance = this->moving_variance.clone();
        }
        value::Tensor NormalizationImpl::calculate(const value::Tensor& input)
        {
            value::Tensor normal;
            if (tensor_array::value::use_grad)
            {
                value::Tensor temp_mean = input.mean(this->dims_mean);
                value::Tensor temp_variance = input.variance(this->dims_mean);
                normal = (input - temp_mean) / power(add(temp_variance, value::values(input.get_buffer().shape(), this->eps)), value::values(input.get_buffer().shape(), .5f));
                this->moving_mean *= value::values(this->moving_mean.get_buffer().shape(), 1.f - momentum);
                this->moving_mean += (value::values(temp_mean.get_buffer().shape(), momentum) * temp_mean);
                this->moving_mean = this->moving_mean.clone();
                this->moving_variance *= value::values(this->moving_variance.get_buffer().shape(), 1.f - momentum);
                this->moving_variance += (value::values(temp_variance.get_buffer().shape(), momentum) * temp_variance);
                this->moving_variance = this->moving_variance.clone();
            }
            else
                normal = (input - this->moving_mean) / power(add(this->moving_variance, value::values(input.get_buffer().shape(), this->eps)), value::values(input.get_buffer().shape(), .5f)).clone();
            return gamma * normal + beta;
        }
        NormalizationImpl::NormalizationImpl(const std::initializer_list<unsigned char>& dims_mean, float eps, float momentum) :
            eps(eps),
            momentum(momentum),
            dims_mean(dims_mean),
            moving_mean(),
            moving_variance(),
            gamma(),
            beta()
        {
            this->map_tensor.insert(std::make_pair("gamma", &gamma));
            this->map_tensor.insert(std::make_pair("beta", &beta));
            this->map_tensor.insert(std::make_pair("moving_mean", &moving_mean));
            this->map_tensor.insert(std::make_pair("moving_variance", &moving_variance));
        }
        NormalizationImpl::~NormalizationImpl()
        {
        }
	}
}