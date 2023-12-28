#include "linear.hh"

namespace tensor_array
{
	namespace layers
	{
        LinearImpl::LinearImpl(unsigned int units) :
            weight(),
            bias(value::values({ units }, 0.f))
        {
            this->map_tensor.insert(std::make_pair("bias", &bias));
            this->map_tensor.insert(std::make_pair("weight", &weight));
        }

        void LinearImpl::init_value(const value::Tensor& input)
        {
            auto& buf = input.get_buffer();
            if (!weight.has_tensor())
                weight = value::values({ buf.shape().end()[-1], bias.get_buffer().shape().begin()[0] }, 1.f / buf.shape().end()[-1]).tensor_cast(buf.type());
        }

        value::Tensor LinearImpl::calculate(const value::Tensor& input)
        {
            return matmul(input, weight) + bias;
        }
	}
}