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

#include <algorithm>
#include <mutex>
#include "tensorbase.hh"

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

#define USING_DATA_TYPE_FLOAT (float)(double)
#define USING_DATA_TYPE_SINT (int8_t)(int16_t)(int32_t)(int64_t)
#define USING_DATA_TYPE_UINT (uint8_t)(uint16_t)(uint32_t)(uint64_t)
#define USING_DATA_TYPE USING_DATA_TYPE_SINT USING_DATA_TYPE_UINT USING_DATA_TYPE_FLOAT

#define LOOP(seq) END(A seq)
#define BODY(x) ADD_CODE(x)
#define A(x) BODY(x) B
#define B(x) BODY(x) A
#define A_END
#define B_END
#define END(...) END_(__VA_ARGS__)
#define END_(...) __VA_ARGS__##_END

namespace tensor_array
{
	namespace value
	{
        extern CUDA_ML_API bool use_grad;

#ifdef TENSOR_CONTENT
        void* create_mem_101(std::size_t s, const void* dat);
        class DataBuffer
        {
        public:
            template<typename T>
            constexpr DataBuffer(const T(&data)) :
                data(create_mem_101(sizeof(T), &data)),
                data_size(sizeof(T))
            {
                static_assert(std::is_trivially_copyable_v<T>, "Requied default constructor");
            }
            template<typename T>
            constexpr DataBuffer(const std::initializer_list<T> &data) :
                data(create_mem_101(sizeof(T) * data.size(), data.begin())),
                data_size(sizeof(T) * data.size())
            {
                static_assert(std::is_trivially_copyable_v<T>, "Requied default constructor");
            }
            DataBuffer();
            DataBuffer(std::nullptr_t);
            DataBuffer(const DataBuffer&);
            ~DataBuffer();
            const void* const& get_data() const;
            const std::size_t& get_data_size() const;
            DataBuffer& operator=(const DataBuffer&);
            friend bool operator==(const DataBuffer&, const DataBuffer&);
        private:
            const void* data;
            std::size_t data_size;
        };

        class Derivation;
#endif

        struct dimension
        {
            unsigned int x = 1U, y = 1U, z = 1U;
        };

        struct ConvolutionParameter
        {
            dimension
                input,
                kernel,
                strides,
                dilation;
        };

        /**
         * \brief Dynamic derivative tensor.
         * \brief This class use to calculate the tensor.
         */
        class CUDA_ML_API Tensor
        {
        public:
            /**
             * \brief Create an empty tensor.
             */
            constexpr Tensor() = default;

            /**
             * \brief Base Tensor copy.
             */
            Tensor(const TensorBase&);

            /**
             * \brief Base Tensor move.
             */
            Tensor(TensorBase&&);

            ~Tensor();

            friend class WeakTensor;
            friend struct std::hash<tensor_array::value::Tensor>;
            friend struct std::equal_to<tensor_array::value::Tensor>;

            /**
             * \brief This class can iterate copy child tensor by index and derivate to parent tensor,
             */
            class CUDA_ML_API Iterator
            {
            public:
                using iterator_category = std::forward_iterator_tag;
                using difference_type = std::ptrdiff_t;
                using value_type = Tensor;
                using reference = value_type;
                using reference_left = const value_type&;
                Iterator(reference_left, unsigned int);
                reference operator*() const;
                Iterator& operator++();
                Iterator& operator--();
                Iterator operator++(int);
                Iterator operator--(int);
                friend bool CUDA_ML_API operator==(const Iterator&, const Iterator&);
                friend bool CUDA_ML_API operator!=(const Iterator&, const Iterator&);
            private:
                unsigned long long index;
                reference_left ref;
            };
            struct Slice
            {
                int begin = 0;
                int end = -1;
                int strides = 1;
            };
            Tensor clone() const;
            void save(const char*) const;
            int real_index(int) const;
            Slice correct_slice(const Slice&) const;
            bool& multithread_derive();
            long tensor_use_count();
            void calc_grad();
            Iterator begin() const;
            Iterator end() const;
            Iterator cbegin() const;
            Iterator cend() const;
            const TensorBase& get_buffer() const;
            Tensor padding(unsigned int);
            Tensor loss(const Tensor&) const;
            Tensor mean(unsigned char) const;
            Tensor variance(unsigned char) const;
            Tensor mean(const std::initializer_list<unsigned char>&) const;
            Tensor variance(const std::initializer_list<unsigned char>&) const;
            Tensor mean(const std::vector<unsigned char>&) const;
            Tensor variance(const std::vector<unsigned char>&) const;
            Tensor value_scalar() const;
            Tensor get_grad() const;
            Tensor expand(unsigned char, unsigned int);
            Tensor reshape(const std::initializer_list<unsigned int>&) const;
            Tensor reshape(const std::vector<unsigned int>&) const;
            Tensor tensor_cast(const std::type_info&) const;
            Tensor conv_padding(const dimension&) const;
#ifdef TENSOR_CONTENT
            friend Tensor derive_transpose(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor derive_reshape_cast(const Tensor&, const Tensor&, bool, const DataBuffer&);
#endif
            Tensor transpose(unsigned char, unsigned char) const;
            std::pair<Tensor, Tensor> max(unsigned char = 0) const;
            std::pair<Tensor, Tensor> min(unsigned char = 0) const;
            friend std::pair<Tensor, Tensor> tensor_broadcasting(const Tensor&, const Tensor&, unsigned char, unsigned char);
#ifdef TENSOR_CONTENT
            friend CUDA_ML_API Tensor add_dim(const std::vector<Tensor>&);
#endif
            bool has_tensor() const;
            template<typename T>
            operator T () const;


            Tensor unslice(const std::initializer_list<unsigned int>&, const std::initializer_list<Slice>&) const;

            /**
             * \brief Array Operator.
             * You can chain tensor array operator to a scalar.
             * \param pos Position of this tensor.
             * \return
             * Tensor
             */
            Tensor operator[](unsigned int) const;

            Tensor operator[](const std::initializer_list<Slice>&) const;

            Tensor operator+() const;

            Tensor operator-() const;

            Tensor& operator+=(const Tensor&);

            Tensor& operator-=(const Tensor&);

            Tensor& operator*=(const Tensor&);

            Tensor& operator/=(const Tensor&);

            friend CUDA_ML_API Tensor operator>(const Tensor&, const Tensor&);
            friend CUDA_ML_API Tensor operator<(const Tensor&, const Tensor&);
            friend CUDA_ML_API Tensor operator&&(const Tensor&, const Tensor&);
            friend CUDA_ML_API Tensor operator||(const Tensor&, const Tensor&);
            Tensor operator!();
            Tensor exp() const;
            Tensor sin() const;
            Tensor cos() const;
            Tensor tan() const;
            Tensor sinh() const;
            Tensor cosh() const;
            Tensor tanh() const;
            Tensor sigmoid() const;
            Tensor reduce_sum(unsigned char) const;
            Tensor reduce_max(unsigned char) const;
            Tensor reduce_min(unsigned char) const;

            Tensor log() const;
#ifdef TENSOR_CONTENT
            friend Tensor tensor_rand(const std::initializer_list<unsigned int>&, unsigned int);
            
            friend Tensor add(const Tensor&, const Tensor&, bool);

            friend Tensor power(const Tensor&, const Tensor&, bool);

            friend Tensor multiply(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor divide(const Tensor&, const Tensor&, bool);

            friend Tensor dot(const Tensor&, const Tensor&, bool, const DataBuffer&);

            friend Tensor condition(const Tensor&, const Tensor&, const Tensor&, bool);

            friend Tensor derive_convolution_padding(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_padding(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_col2im(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor convolution_im2col(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor matmul(const Tensor&, const Tensor&, bool, const DataBuffer&);
            friend Tensor batchedmatmul(const Tensor& a, const Tensor& b, bool is_derive, const DataBuffer&);

            Tensor tensor_cast(const std::type_info&, bool) const;
#endif
            friend CUDA_ML_API std::ostream& operator<<(std::ostream&, const Tensor&);

        private:
#ifdef TENSOR_CONTENT
            friend class TensorContentDerivation;
            friend class TensorContentGradient;
            friend struct Derivation;
            Tensor slice(const std::initializer_list<Slice>&, bool) const;
            Tensor reshape(const std::initializer_list<unsigned int>&, bool) const;
            Tensor transpose(unsigned char, unsigned char, bool) const;
            Tensor exp(bool) const;
            Tensor sin(bool) const;
            Tensor cos(bool) const;
            Tensor tan(bool) const;
            Tensor sinh(bool) const;
            Tensor cosh(bool) const;
            Tensor tanh(bool) const;
            Tensor sigmoid(bool) const;
            template <typename T>
            Tensor cast(bool) const;
            Tensor convolution_convert(const ConvolutionParameter&);
            Tensor(const TensorBase&, const std::vector<std::pair<Tensor, Derivation>>&);
            Tensor(TensorBase&&, std::vector<std::pair<Tensor, Derivation>>&&);
#endif // TENSOR_CONTENT
            struct TensorContent;
            Tensor(const std::shared_ptr<TensorContent>&);
            Tensor(std::shared_ptr<TensorContent>&&);
            std::shared_ptr<TensorContent> tensor_data;
        };

        class CUDA_ML_API WeakTensor
        {
        public:
            WeakTensor(const Tensor&);
            bool is_tensor_expired();
            Tensor to_tensor();
        private:
            std::weak_ptr<Tensor::TensorContent> tensor_data;
        };

        CUDA_ML_API dimension operator+(const dimension&, const dimension&);

        CUDA_ML_API dimension operator-(const dimension&, const dimension&);

        CUDA_ML_API dimension operator*(const dimension&, const dimension&);

        CUDA_ML_API dimension operator/(const dimension&, const dimension&);

        /**
         * \brief Plus 2 n-d tensors.
         * \param other The tensor that plus with this.
         * \return
         * Tensor
         */
        CUDA_ML_API Tensor operator+(const Tensor&, const Tensor&);

        CUDA_ML_API Tensor operator-(const Tensor&, const Tensor&);

        /**
         * \brief Multiply 2 n-d tensors.
         * \param other The tensor that multiply with this.
         * \return
         * Tensor
         */
        CUDA_ML_API Tensor operator*(const Tensor&, const Tensor&);

        CUDA_ML_API Tensor operator/(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor operator!=(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor operator==(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor operator>=(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor operator<=(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor tensor_file_load(const char*);
        CUDA_ML_API Tensor power(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor add(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor multiply(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor divide(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor dot(const Tensor&, const Tensor&);
        /**
         * \brief Matrix multiplication 2 matrices.
         * \param a Matrix/Tensor that has size (batch*)m*k.
         * \param b Matrix/Tensor that has size (batch*)k*n.
         * \return Tensor - Matrix that has size (batch*)m*n.
         * \exception a.col != b.row 
         */
        CUDA_ML_API Tensor matmul(const Tensor&, const Tensor&);
        CUDA_ML_API Tensor condition(const Tensor&, const Tensor&, const Tensor&);
        /**
         * \brief Convolution
         * \brief Only suport 1D, 2D, 3D convolution
         * \param input Tensor (N, C, ...).
         * \param kernel Tensor (C, ..., K).
         * \param strides dimension.
         * \param dilation dimension.
         * \return
         * Tensor (N, K, ...)
         */
        CUDA_ML_API Tensor convolution(const Tensor&, const Tensor&, const dimension& = value::dimension(), const dimension& = value::dimension());
        CUDA_ML_API std::pair<Tensor, Tensor> tensor_broadcasting(const Tensor&, const Tensor&, unsigned char = 0, unsigned char = 0);
        CUDA_ML_API Tensor tensor_rand(const std::initializer_list<unsigned int>&, unsigned int = std::rand());
#define ADD_CODE(TYPE) CUDA_ML_API Tensor values(const std::initializer_list<unsigned int>&, TYPE);
        LOOP(USING_DATA_TYPE);
#undef ADD_CODE
#ifndef TENSOR_CONTENT
        CUDA_ML_API Tensor add_dim(const std::vector<Tensor>&);
#endif
        CUDA_ML_API const std::type_info& comparison_type(const std::type_info&, const std::type_info&);
        CUDA_ML_API Tensor tensor_rand(const std::vector<unsigned int>&, unsigned int = std::rand());

#ifdef TENSOR_CONTENT
        class Derivation
        {
        private:
            typedef Tensor(*multiply_type)(const Tensor&, const Tensor&, bool, const DataBuffer&);
            Tensor derive_value;
            multiply_type multi;
            bool is_value_before;
            DataBuffer option;
        public:
            Derivation(const Tensor&, const multiply_type, bool = false, const DataBuffer & = DataBuffer());
            Tensor calc_grad_temp(const Tensor&) const;
            friend std::vector<Derivation> check_derive_data(const std::vector<Derivation>&);
        };
#endif

        template<typename T>
        Tensor::operator T () const
        {
            const TensorBase& base = this->get_buffer();
            if (base.shape().size() != 0 && base.type() != typeid(T)) throw 0;
            return *static_cast<const T*>(base.data());
        }

        template<typename T>
        inline Tensor values(const std::vector<unsigned int>& shape_vector, T value)
        {
            return values(wrapper::initializer_wrapper<unsigned int>(shape_vector.begin().operator->(), shape_vector.end().operator->()), value);
        }

        template<typename T>
        inline Tensor zeros(const std::initializer_list<unsigned int>& shape_list)
        {
            return TensorBase(typeid(T), shape_list);
        }

        template<typename T>
        inline Tensor zeros(const std::vector<unsigned int>& shape_vector)
        {
            return zeros<T>(wrapper::initializer_wrapper<unsigned int>(shape_vector.begin().operator->(), shape_vector.end().operator->()));
        }
}
}

template<>
struct std::hash<tensor_array::value::Tensor>
{
    inline std::size_t operator()(const tensor_array::value::Tensor& t) const
    {
        return std::hash<std::shared_ptr<tensor_array::value::Tensor::TensorContent>>()(t.tensor_data);
    }
};

template<>
struct std::equal_to<tensor_array::value::Tensor>
{
    inline std::size_t operator()(const tensor_array::value::Tensor& a, const tensor_array::value::Tensor& b) const
    {
        return std::equal_to<std::shared_ptr<tensor_array::value::Tensor::TensorContent>>()(a.tensor_data, b.tensor_data);
    }
};

#undef LOOP
#undef BODY
#undef A
#undef B
#undef A_END
#undef B_END
#undef END
#undef END_

#undef USING_DATA_TYPE
#undef USING_DATA_TYPE_FLOAT
#undef USING_DATA_TYPE_SINT
#undef USING_DATA_TYPE_UINT

#undef CUDA_ML_API