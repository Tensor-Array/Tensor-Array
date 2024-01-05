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

#include "data_type_wrapper.hh"
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensorbase.hh"
#undef TENSOR_CONTENT
#endif
#include <cstdio>
#include <cstring>

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
        class TensorBuf final : public TensorBase::TensorStorage
        {
        public:
            TensorBuf(const std::type_info&, unsigned char, const unsigned int*, const void*, const devices::Device&, const devices::Device&);
            TensorBuf(const TensorBuf&);
            ~TensorBuf() override;
            const std::type_info& type() const override;
            std::unique_ptr<TensorStorage> clone() const override;
            std::initializer_list<unsigned int> dim_sizes() const override;
            const void* data() const override;
            size_t data_size() const override;
            std::unique_ptr<TensorStorage> child_create(unsigned int) const override;
            const devices::Device& get_device() const override;
        private:
            const void* const dat;
            const unsigned int* const sizes;
            const unsigned char dim;
            const std::type_info& dtype;
            const devices::Device device;
        };

        TensorBase::TensorBase(const std::type_info& dtype, const std::initializer_list<unsigned int>& shape_vec, const void* dat, const devices::Device& dev_other, const devices::Device& dev_this):
            instance(std::make_unique<TensorBuf>(dtype, static_cast<unsigned char>(shape_vec.size()), shape_vec.begin(), dat, dev_other, dev_this))
        {
        }

        TensorBase::TensorBase(const std::type_info& dtype, const std::vector<unsigned int>& shape_vec, const void* dat, const devices::Device& dev_other, const devices::Device& dev_this):
            TensorBase(dtype, wrapper::initializer_wrapper<unsigned int>(shape_vec.begin().operator->(), shape_vec.end().operator->()), dat, dev_other, dev_this)
        {
        }

        TensorBase::TensorBase(const TensorBase& other) :
            instance(other.instance->clone())
        {
        }

        TensorBase::TensorBase(const TensorBase& other, const devices::Device& other_device) :
            TensorBase(other.type(), other.shape(), other.data(), other.get_device(), other_device)
        {
        }

        TensorBase& TensorBase::operator=(const TensorBase& other)
        {
            TensorBase temp(other);
            this->swap(temp);
            return *this;
        }

        TensorBase TensorBase::operator[](unsigned int index) const
        {
            TensorBase temp;
            temp.instance = this->instance->child_create(index);
            return temp;
        }

        const std::type_info& TensorBase::type() const
        {
            return this->instance->type();
        }

        std::initializer_list<unsigned int> TensorBase::shape() const
        {
            return this->instance->dim_sizes();
        }

        const void* TensorBase::data() const
        {
            return this->instance->data();
        }

        size_t TensorBase::data_size() const
        {
            return this->instance->data_size();
        }

        const devices::Device& TensorBase::get_device() const
        {
            return this->instance->get_device();
        }

        TensorBase TensorBase::change_device(const devices::Device& dev) const
        {
            return TensorBase(*this, dev);
        }

        bool TensorBase::has_tensor() const
        {
            return static_cast<bool>(this->instance);
        }

        void TensorBase::swap(TensorBase& other)
        {
            this->instance.swap(other.instance);
        }

        void TensorBase::save(const char* dir) const
        {
            if (static_cast<bool>(this->instance))
            {
                if (std::FILE* tensor_file = std::fopen(dir, "wb"))
                {
                    std::initializer_list<unsigned int> list = this->shape();
                    std::size_t temp_save;
                    std::size_t total_dim_size = 1;
                    for (auto& it : list)
                        total_dim_size *= it;
                    datatype::DataType temp_type = datatype::warp_type(this->type());
                    unsigned char temp_dim_size = list.size();
                    temp_save = std::fwrite(&temp_type, sizeof(datatype::DataType), 1, tensor_file);
                    temp_save = std::fwrite(&temp_dim_size, sizeof(char), 1, tensor_file);
                    temp_save = std::fwrite(list.begin(), sizeof(unsigned int), list.size(), tensor_file);
                    temp_save = std::fwrite(this->data(), get_sizeof_type(this->type()), total_dim_size, tensor_file);
                    std::fclose(tensor_file);
                }
            }
        }

		const unsigned int* create_arr_dim_sizes(size_t other_dim, const unsigned int* other_sizes)
		{
			unsigned int* temp_sizes = new unsigned int[other_dim];
			if (other_sizes)
				std::memcpy(temp_sizes, other_sizes, other_dim * sizeof(unsigned int));
			return temp_sizes;
		}

		void* create_data_1(size_t other_data_size, const void* other_data, const devices::Device& dev_other, const devices::Device& dev_this)
		{
			void* temp_sizes = operator new(other_data_size, dev_this);
            if (other_data)
                devices::device_memcpy(temp_sizes, dev_this, other_data, dev_other, other_data_size);
            else
                devices::device_memset(temp_sizes, dev_this, 0, other_data_size);
			return temp_sizes;
		}

        std::size_t get_sizeof_type(const std::type_info& t_info)
        {
            return dynamic_type_size[t_info];
        }

		unsigned long long get_buffer_size0(const std::type_info& dtype, unsigned char dim, const unsigned int* sizes)
		{
			unsigned long long temp_size = 1;
			for (unsigned char i = 0; i < dim; i++)
				temp_size *= sizes[i];
			return temp_size * get_sizeof_type(dtype);
		}

        TensorBuf::TensorBuf(const std::type_info& dtype, unsigned char dim, const unsigned int* sizes, const void* dat, const devices::Device& dev_other, const devices::Device& dev_this) :
            dtype(dtype),
            dim(dim),
            sizes(create_arr_dim_sizes(dim, sizes)),
			dat(create_data_1(get_buffer_size0(dtype, dim, sizes), dat, dev_other, dev_this)),
            device(dev_this)
        {
            if (dtype == typeid(void)) throw 0;
        }

        TensorBuf::TensorBuf(const TensorBuf& other) :
            TensorBuf(other.dtype, other.dim, other.sizes, other.dat, other.device, other.device)
        {
        }

        TensorBuf::~TensorBuf()
        {
            delete[] this->sizes;
            operator delete(const_cast<void*>(this->dat), this->device);
        }

        const std::type_info& TensorBuf::type() const
        {
            return this->dtype;
        }

        std::unique_ptr<TensorBase::TensorStorage> TensorBuf::clone() const
        {
            return std::make_unique<TensorBuf>(*this);
        }

        std::initializer_list<unsigned int> TensorBuf::dim_sizes() const
        {
            return wrapper::initializer_wrapper<unsigned int>(this->sizes, this->sizes + this->dim);
        }

        const void* TensorBuf::data() const
        {
            return this->dat;
        }

        size_t TensorBuf::data_size() const
        {
            return get_buffer_size0(this->dtype, this->dim, this->sizes);
        }
        std::unique_ptr<TensorBase::TensorStorage> TensorBuf::child_create(unsigned int index) const
        {
            if (this->dim == 0) throw std::exception();
            const void* data_ptr = reinterpret_cast<const void*>(reinterpret_cast<std::size_t>(this->dat) + index * (this->data_size() / this->sizes[0]));
            return std::make_unique<TensorBuf>(this->dtype, this->dim - 1, &this->sizes[1], data_ptr, this->device, this->device);
        }
        const devices::Device& TensorBuf::get_device() const
        {
            return this->device;
        }
    }
}

#undef LOOP
#undef BODY
#undef A
#undef B
#undef A_END
#undef B_END
#undef END
#undef END_
