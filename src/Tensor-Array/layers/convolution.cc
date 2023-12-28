#include "convolution.hh"

namespace tensor_array
{
    namespace layers
    {
        value::Tensor conv_bias_init(unsigned int dim, unsigned int filter)
        {
            std::vector<unsigned int> bias_shape;
            bias_shape.insert(bias_shape.end(), filter);
            constexpr unsigned int one = 1U;
            bias_shape.insert(bias_shape.end(), dim, one);
            return tensor_array::value::values(bias_shape, 0.f);
        }

        value::dimension padding_resize(const value::dimension& padding, unsigned char dim)
        {
            return
            {
                (dim > 0) ? padding.x : 0U,
                (dim > 1) ? padding.y : 0U,
                (dim > 2) ? padding.z : 0U
            };
        }

        ConvolutionLayerImpl::ConvolutionLayerImpl(unsigned char dim, const value::dimension& kernel_size, unsigned int filter, const value::dimension& strides, const value::dimension& padding, const value::dimension& dilation) :
            dim(dim),
            bias(conv_bias_init(dim, filter)),
            kernel_size(kernel_size),
            filter(filter),
            strides(strides),
            padding(padding_resize(padding, dim)),
            dilation(dilation)
        {
            this->map_tensor.insert(std::make_pair("bias", &bias));
            this->map_tensor.insert(std::make_pair("kernel", &kernel));
        }

        value::Tensor ConvolutionLayerImpl::calculate(const value::Tensor& input)
        {
            return convolution(input.conv_padding(this->padding), this->kernel, this->strides, this->dilation) + this->bias;
        }

        Conv1D_Impl::Conv1D_Impl(const value::dimension& kernel_size, unsigned int filter, const value::dimension& strides, const value::dimension& dilation):
            ConvolutionLayerImpl(1, kernel_size, filter, strides, dilation)
        {
        }

        void Conv1D_Impl::init_value(const value::Tensor& input)
        {
            if (!this->kernel.has_tensor())
                this->kernel = value::values({ input.get_buffer().shape().begin()[1], this->kernel_size.x, this->filter }, 1.f / (input.get_buffer().shape().begin()[1] * this->kernel_size.x));
        }
        Conv2D_Impl::Conv2D_Impl(const value::dimension& kernel_size, unsigned int filter, const value::dimension& strides, const value::dimension& dilation) :
            ConvolutionLayerImpl(2, kernel_size, filter, strides, dilation)
        {
        }
        void Conv2D_Impl::init_value(const value::Tensor& input)
        {
            if (!this->kernel.has_tensor())
                this->kernel = value::values
                (
                    {
                        input.get_buffer().shape().begin()[1],
                        this->kernel_size.x,
                        this->kernel_size.y,
                        this->filter
                    }, 1.f / (input.get_buffer().shape().begin()[1] * this->kernel_size.x * this->kernel_size.y)
                );
        }

        Conv3D_Impl::Conv3D_Impl(const value::dimension& kernel_size, unsigned int filter, const value::dimension& strides, const value::dimension& dilation) :
            ConvolutionLayerImpl(3, kernel_size, filter, strides, dilation)
        {
        }

        void Conv3D_Impl::init_value(const value::Tensor& input)
        {
            if (!this->kernel.has_tensor())
                this->kernel = value::values
                (
                    {
                        input.get_buffer().shape().begin()[1],
                        this->kernel_size.x,
                        this->kernel_size.y,
                        this->kernel_size.z,
                        this->filter
                    }, 1.f / (input.get_buffer().shape().begin()[1] * this->kernel_size.x * this->kernel_size.y * this->kernel_size.z)
                );
        }
    }
}

