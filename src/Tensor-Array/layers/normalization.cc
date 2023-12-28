#include "normalization.hh"

namespace tensor_array
{
	namespace layers
	{
        void NormalizationImpl::init_value(const value::Tensor& input)
        {
            std::vector<unsigned int> temp = input.get_buffer().shape();
            const std::type_info& type = input.get_buffer().type();
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
            this->moving_mean = this->moving_mean.new_grad_copy();
            this->moving_variance = this->moving_variance.new_grad_copy();
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
                this->moving_mean = this->moving_mean.new_grad_copy();
                this->moving_variance *= value::values(this->moving_variance.get_buffer().shape(), 1.f - momentum);
                this->moving_variance += (value::values(temp_variance.get_buffer().shape(), momentum) * temp_variance);
                this->moving_variance = this->moving_variance.new_grad_copy();
            }
            else
                normal = (input - this->moving_mean) / power(add(this->moving_variance, value::values(input.get_buffer().shape(), this->eps)), value::values(input.get_buffer().shape(), .5f)).new_grad_copy();
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