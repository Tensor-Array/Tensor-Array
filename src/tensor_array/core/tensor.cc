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

#include <cassert>
#include <thread>
#include <forward_list>
#include <ostream>
#include <cstring>
#include "data_type_wrapper.hh"
#ifndef TENSOR_CONTENT
#define TENSOR_CONTENT
#include "tensor.hh"
#undef TENSOR_CONTENT
#endif // !TENSOR_CONTENT
#include <unordered_set>

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
        bool use_grad = true;

        class Tensor::TensorContent
        {
        private:
            const TensorBase buf;
        protected:
            std::mutex tensor_mutex;
            std::unordered_set<const Tensor::TensorContent*> forward_back;
            TensorBase grad;
        public:
            friend const TensorBase& Tensor::get_buffer() const;
            TensorContent() = delete;
            TensorContent(const TensorBase&);
            TensorContent(TensorBase&&);
            TensorContent(const TensorContent&) = delete;
            TensorContent(TensorContent&&) = delete;
            virtual ~TensorContent() = default;
            virtual void reset_grad();
            virtual void calc_grad(const Tensor&);
            void reset_grad_thread(const Tensor* tensor_ptr);
            void calc_grad_thread(Tensor grad, const std::pair<const Tensor, std::vector<Derivation>>* data_ptr);
            Tensor get_grad() const;
        };

        class TensorContentDerivation final : public Tensor::TensorContent
        {
        private:
            const std::unordered_map<Tensor, std::vector<Derivation>> derive_data;
            const bool can_calc_grad = use_grad;
            bool derive_multithread = 0;
        public:
            TensorContentDerivation(const TensorBase&, const std::vector<std::pair<Tensor, Derivation>>&);
            TensorContentDerivation(TensorBase&&, std::vector<std::pair<Tensor, Derivation>>&&);
            void reset_grad() override;
            void calc_grad(const Tensor&) override;
            bool& multithread_derive();
        };

        Derivation::Derivation(const Tensor& derive_value, const multiply_type multi, bool is_value_before, const DataBuffer& option) :
            derive_value(derive_value),
            multi(multi),
            is_value_before(is_value_before),
            option(option)
        {
        }

        std::unordered_map<Tensor, std::vector<Derivation>> check_derive_data(const std::vector<std::pair<Tensor, Derivation>>& derive_data)
        {
            std::unordered_map<Tensor, std::vector<Derivation>> better_derive_data;
            for (auto& [t, derive] : derive_data)
                better_derive_data[t].push_back(derive);
            return better_derive_data;
        }

        Tensor Derivation::calc_grad_temp(const Tensor& grad) const
        {
            return (this->is_value_before) ? this->multi(this->derive_value, grad, false, this->option) : this->multi(grad, this->derive_value, false, this->option);
        }

        bool& TensorContentDerivation::multithread_derive()
        {
            return this->derive_multithread;
        }

        Tensor Tensor::TensorContent::get_grad() const
        {
            return (this->grad.has_tensor()) ? Tensor(this->grad) : Tensor();
        }

        void Tensor::TensorContent::reset_grad_thread(const Tensor* tensor_ptr)
        {
            if (tensor_ptr->tensor_data->forward_back.find(this) == tensor_ptr->tensor_data->forward_back.end())
            {
                tensor_ptr->tensor_data->reset_grad();
                tensor_ptr->tensor_data->forward_back.insert(this);
            }
        }

        void TensorContentDerivation::reset_grad()
        {
            this->TensorContent::reset_grad();
            std::forward_list<std::thread> thread_list;
            for (auto& [t, _] : this->derive_data)
                if (this->derive_multithread)
                    thread_list.push_front(std::thread(&TensorContent::reset_grad_thread, this, &t));
                else
                    this->reset_grad_thread(&t);
            for (std::thread& thread_item : thread_list)
                if (thread_item.joinable())
                    thread_item.join();
        }

        void Tensor::TensorContent::reset_grad()
        {
            std::lock_guard tensor_lock(this->tensor_mutex);
            this->grad = zeros<int>(this->buf.shape()).tensor_cast(this->buf.type()).get_buffer();
        }

        void Tensor::TensorContent::calc_grad_thread(Tensor grad, const std::pair<const Tensor, std::vector<Derivation>>* data_ptr)
        {
            auto& [child, derive_data] = *data_ptr;
            Tensor temp_sum = zeros<int>(child.get_buffer().shape()).tensor_cast(child.get_buffer().type(), false);
            for (auto& it : derive_data)
                temp_sum = add(temp_sum, it.calc_grad_temp(grad), false);
            child.tensor_data->forward_back.erase(this);
            child.tensor_data->calc_grad(temp_sum);
        }

        void TensorContentDerivation::calc_grad(const Tensor& grad)
        {
            this->TensorContent::calc_grad(grad);
            if (this->can_calc_grad && this->forward_back.empty())
            {
                std::lock_guard tensor_lock(this->tensor_mutex);
                std::forward_list<std::thread> thread_list;
                for (auto& dat : this->derive_data)
                    if (this->derive_multithread)
                        thread_list.push_front(std::thread(&Tensor::TensorContent::calc_grad_thread, this, this->grad, &dat));
                    else
                        this->calc_grad_thread(this->grad, &dat);
                for (std::thread& thread_item : thread_list)
                    if (thread_item.joinable())
                        thread_item.join();
            }
        }

        void Tensor::TensorContent::calc_grad(const Tensor& grad)
        {
            std::lock_guard tensor_lock(this->tensor_mutex);
            this->grad = add(this->grad, grad, false).get_buffer();
        }

        Tensor derive_dim_added(const Tensor& children, const Tensor&, bool, const DataBuffer& dat)
        {
            return children[*static_cast<const unsigned int*>(dat.get_data())];
        }

        Tensor add_dim(const std::vector<Tensor>& list)
        {
            std::vector<std::pair<Tensor, Derivation>> derive_list;
            void* content = operator new(list.size() * list[0].get_buffer().data_size(), devices::local_device());
            std::vector<unsigned int> last_sizes = list[0].get_buffer().shape();
            const std::type_info& last_type = list[0].get_buffer().type();
            for (unsigned int i = 0; i < list.size(); i++)
            {
                if (std::vector<unsigned int>(list[i].get_buffer().shape()) != last_sizes && list[i].get_buffer().type() != last_type) throw 0;
                derive_list.push_back(std::make_pair(list[i], Derivation(Tensor(), derive_dim_added, false, i)));
                void* temp_ptr = reinterpret_cast<void*>(reinterpret_cast<unsigned long long>(content) + (i * list[i].get_buffer().data_size()));
                devices::device_memcpy(temp_ptr, devices::local_device(), list[i].get_buffer().data(), list[i].get_buffer().get_device(), list[i].get_buffer().data_size());
            }
            last_sizes.insert(last_sizes.begin(), list.size());
            TensorBase other_buf(list[0].get_buffer().type(), last_sizes, content);
            operator delete(content, devices::local_device());
            return Tensor(std::move(other_buf), std::move(derive_list));
        }

        std::pair<Tensor, Tensor> Tensor::max(unsigned char dim) const
        {
            assert(this->get_buffer().get_device().dev_t == devices::CPU);
            if (dim)
            {
                std::vector<Tensor> temp_tensors;
                std::vector<Tensor> temp_dims;
                for (auto it : *this)
                {
                    const std::pair<Tensor, Tensor> temp = it.max(dim - 1);
                    temp_tensors.push_back(temp.first);
                    temp_dims.push_back(temp.second);
                }
                return std::make_pair(add_dim(temp_tensors), add_dim(temp_dims));
            }
            if (this->get_buffer().shape().size() == 0)
                return std::make_pair(*this, Tensor());
            auto [temp_tensor, temp_dim] = this->operator[](0).max(dim);
            unsigned int max_i = 0;
            for (unsigned int i = 1; i < *this->get_buffer().shape().begin(); i++)
            {
                const std::pair<Tensor, Tensor> temp = this->operator[](i).max(dim);
                assert(temp.first.get_buffer().shape().size() == 0);
                bool temp_check_data_type = false;
#define ADD_CODE(TEMP)\
if (temp.first.get_buffer().type() == typeid(TEMP) && temp_tensor.get_buffer().type() == typeid(TEMP))\
temp_check_data_type = TEMP(temp.first) > TEMP(temp_tensor);
                LOOP(USING_DATA_TYPE);
#undef ADD_CODE
                if (temp_check_data_type || (!temp_tensor.has_tensor() && !temp_dim.has_tensor()))
                {
                    temp_tensor = temp.first;
                    temp_dim = temp.second;
                    max_i = i;
                }
            }
            bool test_tensor = temp_dim.has_tensor() && temp_dim.get_buffer().shape().begin() != temp_dim.get_buffer().shape().end();
            unsigned int dim_size1 = test_tensor ? *temp_dim.get_buffer().shape().begin() : 0U;
            unsigned int* dims_ptr = new unsigned int[1 + dim_size1] {max_i};
            if (test_tensor)
                std::memcpy(dims_ptr + 1U, temp_dim.get_buffer().data(), dim_size1);
            TensorBase base_dim(typeid(unsigned int), {dim_size1 + 1}, dims_ptr, devices::DEVICE_CPU_0);
            delete[] dims_ptr;
            return std::make_pair(temp_tensor, base_dim);
        }

        std::pair<Tensor, Tensor> Tensor::min(unsigned char dim) const
        {
            assert(this->get_buffer().get_device().dev_t == devices::CPU);
            if (dim)
            {
                std::vector<Tensor> temp_data;
                std::vector<Tensor> temp_dim;
                for (auto it : *this)
                {
                    const std::pair<Tensor, Tensor> temp = it.max(dim - 1);
                    temp_data.push_back(temp.first);
                    temp_dim.push_back(temp.second);
                }
                return std::make_pair(add_dim(temp_data), add_dim(temp_dim));
            }
            if (this->get_buffer().shape().size() == 0)
                return std::make_pair(*this, Tensor());
            auto [temp_tensor, temp_dim] = this->operator[](0).max(dim);
            unsigned int max_i = 0;
            for (unsigned int i = 1; i < *this->get_buffer().shape().begin(); i++)
            {
                const std::pair<Tensor, Tensor> temp = this->operator[](i).max(dim);
                assert(temp.first.get_buffer().shape().size() == 0);
                bool temp_check_data_type = false;
#define ADD_CODE(TEMP)\
if (temp.first.get_buffer().type() == typeid(TEMP) && temp_tensor.get_buffer().type() == typeid(TEMP))\
temp_check_data_type = TEMP(temp.first) < TEMP(temp_tensor);
                LOOP(USING_DATA_TYPE);
#undef ADD_CODE
                if (temp_check_data_type || (!temp_tensor.has_tensor() && !temp_dim.has_tensor()))
                {
                    temp_tensor = temp.first;
                    temp_dim = temp.second;
                    max_i = i;
                }
            }
            bool test_tensor = temp_dim.has_tensor() && temp_dim.get_buffer().shape().begin() != temp_dim.get_buffer().shape().end();
            unsigned int dim_size1 = test_tensor ? *temp_dim.get_buffer().shape().begin() : 0U;
            unsigned int* dims_ptr = new unsigned int[1 + dim_size1] {max_i};
            if (test_tensor)
                std::memcpy(dims_ptr + 1U, temp_dim.get_buffer().data(), dim_size1);
            TensorBase base_dim(typeid(unsigned int), { dim_size1 + 1 }, dims_ptr, devices::DEVICE_CPU_0);
            delete[] dims_ptr;
            return std::make_pair(temp_tensor, base_dim);
        }

        bool operator==(const Tensor::Iterator& a, const Tensor::Iterator& b)
        {
            return (&a.ref == &b.ref) && (a.index == b.index);
        }

        bool operator!=(const Tensor::Iterator& a, const Tensor::Iterator& b)
        {
            return (&a.ref != &b.ref) || (a.index != b.index);
        }

        std::pair<Tensor, Tensor> tensor_broadcasting(const Tensor& a, const Tensor& b, unsigned char begin_dim, unsigned char end_dim)
        {
            Tensor temp_a(a);
            Tensor temp_b(b);
            unsigned int max_dim = std::max(a.get_buffer().shape().size(), b.get_buffer().shape().size());
            const std::type_info& boardcasting_type = comparison_type(a.get_buffer().type(), b.get_buffer().type());
            if (temp_a.get_buffer().type() != boardcasting_type)
                temp_a.tensor_cast(boardcasting_type);
            if (temp_b.get_buffer().type() != boardcasting_type)
                temp_b.tensor_cast(boardcasting_type);
            while (temp_a.get_buffer().shape().size() != temp_b.get_buffer().shape().size())
            {
                if (temp_a.get_buffer().shape().size() < temp_b.get_buffer().shape().size())
                    temp_a = add_dim({ temp_a });
                if (temp_a.get_buffer().shape().size() > temp_b.get_buffer().shape().size())
                    temp_b = add_dim({ temp_b });
            }
            for (unsigned int i = max_dim - 1 - end_dim; i < max_dim && i >= begin_dim; i--)
                if (temp_a.get_buffer().shape().begin()[i] == temp_b.get_buffer().shape().begin()[i])
                    continue;
                else if (temp_a.get_buffer().shape().begin()[i] == 1)
                    temp_a = temp_a.expand(i, temp_b.get_buffer().shape().begin()[i]);
                else if (temp_b.get_buffer().shape().begin()[i] == 1)
                    temp_b = temp_b.expand(i, temp_a.get_buffer().shape().begin()[i]);
                else
                    throw std::exception();
            return std::make_pair(temp_a, temp_b);
        }

        unsigned long long calculate_size(const std::initializer_list<unsigned int>& list)
        {
            unsigned long long temp = 1;
            for (unsigned int i : list)
                temp *= i;
            return temp;
        }

        Tensor Tensor::unslice(const std::initializer_list<unsigned int>& shape_arr, const std::initializer_list<Slice>& slice_arr) const
        {
            assert(shape_arr.size() == slice_arr.size());
            if (slice_arr.size() == 0)
                return *this;
            std::vector<unsigned int> temp_shape_arr = this->get_buffer().shape();
            std::copy(std::cbegin(shape_arr), std::cend(shape_arr), temp_shape_arr.begin());
            temp_shape_arr.erase(temp_shape_arr.begin());
            std::vector<Tensor> temp_tensors(shape_arr.begin()[0]);
            const Slice slice_begin = this->correct_slice(slice_arr.begin()[0]);
            Tensor temp_zeros = zeros<int>(temp_shape_arr).tensor_cast(this->get_buffer().type());
            for (auto& it : temp_tensors)
                it = temp_zeros;
            unsigned int index = (slice_begin.strides < 0 ? shape_arr.begin()[0] : 0U) + slice_begin.begin;
            for (Tensor it : *this)
            {
                if (index < 0U || index > shape_arr.begin()[0]) break;
                temp_tensors[index] = it.unslice
                (
                    wrapper::initializer_wrapper(shape_arr.begin() + 1, shape_arr.end()),
                    wrapper::initializer_wrapper(slice_arr.begin() + 1, slice_arr.end())
                );
                index += slice_begin.strides;
            }
            return add_dim(temp_tensors).clone();
        }

        Tensor derive_slice(const Tensor& in_value, const Tensor& out_shape, bool, const DataBuffer& dat)
        {
            auto start = static_cast<const Tensor::Slice*>(dat.get_data());
            auto end = static_cast<const Tensor::Slice*>(dat.get_data()) + (dat.get_data_size() / sizeof(Tensor::Slice));
            return in_value.unslice(out_shape.get_buffer().shape(), wrapper::initializer_wrapper(start, end));
        }

        Tensor Tensor::slice(const std::initializer_list<Slice>& slice_arr, bool is_derive) const
        {
            if (slice_arr.size() == 0U)
                return *this;
            std::vector<std::pair<Tensor, Derivation>> temp;
            if (is_derive)
            {
                std::initializer_list<unsigned int> derive_shape = this->get_buffer().shape();
                temp.push_back(std::make_pair(*this, Derivation(zeros<int>(wrapper::initializer_wrapper(derive_shape.begin(), derive_shape.begin() + slice_arr.size())).tensor_cast(this->get_buffer().type()), derive_slice, false, slice_arr)));
            }
            const Slice slice_begin = this->correct_slice(slice_arr.begin()[0]);
            std::vector<Tensor> temp_tensors;
            for (int i = slice_begin.begin; slice_begin.strides < 0 ? i > slice_begin.end : i < slice_begin.end; i += slice_begin.strides)
            {
                Tensor temp_tensor = this->operator[](i);
                temp_tensor = temp_tensor.slice(wrapper::initializer_wrapper(slice_arr.begin() + 1, slice_arr.end()), false);
                temp_tensors.push_back(temp_tensor);
            }
            return Tensor(add_dim(temp_tensors).get_buffer(), temp);
        }

        Tensor::Tensor(const TensorBase& t, const std::vector<std::pair<Tensor, Derivation>>& derive_data) :
            tensor_data(std::make_shared<TensorContentDerivation>(t, derive_data))
        {
        }

        Tensor::Tensor(TensorBase&& t, std::vector<std::pair<Tensor, Derivation>>&& derive_data) :
            tensor_data(std::make_shared<TensorContentDerivation>(std::forward<TensorBase>(t), std::forward<std::vector<std::pair<Tensor, Derivation>>>(derive_data)))
        {
        }

        Tensor::Tensor(const TensorBase& t) :
            tensor_data(std::make_shared<TensorContent>(t))
        {
        }

        Tensor::Tensor(TensorBase&& t):
            tensor_data(std::make_shared<TensorContent>(std::forward<TensorBase>(t)))
        {
        }

        Tensor::Tensor(const std::shared_ptr<TensorContent>& tensor_data):
            tensor_data(tensor_data)
        {
        }
        Tensor::Tensor(std::shared_ptr<TensorContent>&& tensor_data):
            tensor_data(std::forward<std::shared_ptr<TensorContent>>(tensor_data))
        {
        }

        Tensor::~Tensor()
        {
        }

        Tensor Tensor::clone() const
        {
            return Tensor(this->get_buffer());
        }

        void Tensor::save(const char* dir) const
        {
            if (static_cast<bool>(this->tensor_data))
                this->get_buffer().save(dir);
        }

        int Tensor::real_index(int index) const
        {
            const unsigned int& temp_dim_0 = *this->get_buffer().shape().begin();
            return index < 0 ? index + temp_dim_0 : index;
        }

        Tensor::Slice Tensor::correct_slice(const Slice& input_value) const
        {
            return
            {
                this->real_index(input_value.begin),
                this->real_index(input_value.end),
                input_value.strides
            };
        }

        bool& Tensor::multithread_derive()
        {
            TensorContentDerivation* temp = dynamic_cast<TensorContentDerivation*>(this->tensor_data.get());
            assert(temp);
            return temp->multithread_derive();
        }

        long Tensor::tensor_use_count()
        {
            return this->tensor_data.use_count();
        }

        std::mutex calc_grad_mutex;
        void Tensor::calc_grad()
        {
            std::lock_guard calc_grad_lock(calc_grad_mutex);
            this->tensor_data->reset_grad();
            this->tensor_data->calc_grad(values(this->get_buffer().shape(), 1.f).tensor_cast(this->get_buffer().type()));
        }

        Tensor Tensor::transpose(unsigned char dim0, unsigned char dim1, bool is_derive) const
        {
            std::vector<unsigned int> a = this->get_buffer().shape();
            assert(0 <= dim0 && dim0 <= dim1 && dim1 < a.size());
            unsigned int tranpose_shape[4] = { 1U, 1U, a[dim1], 1U};
            for (size_t i = 0; i < dim0; i++)
                tranpose_shape[0] *= a[i];
            for (size_t i = dim0; i < dim1; i++)
                tranpose_shape[1] *= a[i];
            for (size_t i = a.size() - 1U; i > dim1; i--)
                tranpose_shape[3] *= a[i];
            Tensor temp_tensor = *this;
            if (dim0 == dim1) return temp_tensor;
            temp_tensor = derive_transpose(temp_tensor.reshape(wrapper::initializer_wrapper(std::begin(tranpose_shape), std::end(tranpose_shape)), is_derive), Tensor(), is_derive, nullptr);
            if (dim1 - dim0 > 1)
            {
                tranpose_shape[0] *= a[dim1];
                tranpose_shape[2] = tranpose_shape[1] / a[dim0];
                tranpose_shape[1] = a[dim0];
                temp_tensor = derive_transpose(temp_tensor.reshape(wrapper::initializer_wrapper(std::begin(tranpose_shape), std::end(tranpose_shape)), is_derive), Tensor(), is_derive, nullptr);
            }
            unsigned int temp1 = a[dim0];
            a[dim0] = a[dim1];
            a[dim1] = temp1;
            return temp_tensor.reshape(wrapper::initializer_wrapper(a.begin().operator->(), a.end().operator->()), is_derive);
        }

        bool Tensor::has_tensor() const
        {
            return static_cast<bool>(this->tensor_data);
        }

        Tensor Tensor::transpose(unsigned char dim0, unsigned char dim1) const
        {
            return this->transpose(dim0, dim1, true);
        }

        Tensor Tensor::loss(const Tensor& real_value) const
        {
            Tensor temp = values(this->get_buffer().shape(), 2.f);
            return power(*this - real_value, temp).value_scalar();
        }

        Tensor Tensor::mean(unsigned char dim) const
        {
            std::vector<unsigned int> resize = this->get_buffer().shape();
            assert(dim < resize.size());
            unsigned int dim_x = 1;
            for (unsigned char i = 0; i < dim; i++)
                dim_x *= resize[i];
            std::vector<Tensor> arr;
            resize.erase(resize.begin(), std::next(resize.begin(), dim));
            resize.insert(resize.begin(), dim_x);
            Tensor temp_tensor = this->reshape(resize);
            for (Tensor temp1: temp_tensor)
            {
                Tensor temp_loop = zeros<int>(wrapper::initializer_wrapper(temp1.get_buffer().shape().begin() + 1U, temp1.get_buffer().shape().end())).tensor_cast(temp1.get_buffer().type());
                for (Tensor temp2 : temp1)
                    temp_loop = add(temp_loop, temp2);
                temp_loop /= values(temp_loop.get_buffer().shape(), float(this->get_buffer().shape().begin()[dim])).tensor_cast(temp_loop.get_buffer().type());
                arr.push_back(temp_loop);
            }
            resize = this->get_buffer().shape();
            resize[dim] = 1;
            return add_dim(arr).reshape(resize);
        }

        Tensor Tensor::variance(unsigned char dim) const
        {
            return power(*this - this->mean(dim), Tensor(TensorArray<float>{ 1.f })).mean(dim);
        }

        Tensor Tensor::mean(const std::initializer_list<unsigned char>& list_dims) const
        {
            Tensor temp = *this;
            for (auto& it : list_dims)
                temp = temp.mean(it);
            return temp;
        }

        Tensor Tensor::variance(const std::initializer_list<unsigned char>& list_dims) const
        {
            Tensor temp = *this;
            for (auto& it : list_dims)
                temp = temp.variance(it);
            return temp;
        }

        Tensor Tensor::mean(const std::vector<unsigned char>& list_dims) const
        {
            return this->mean(wrapper::initializer_wrapper<unsigned char>(list_dims.begin().operator->(), list_dims.end().operator->()));
        }

        Tensor Tensor::variance(const std::vector<unsigned char>& list_dims) const
        {
            return this->variance(wrapper::initializer_wrapper(list_dims.begin().operator->(), list_dims.end().operator->()));
        }

        Tensor Tensor::value_scalar() const
        {
            return dot(*this, values(this->get_buffer().shape(), 1.f));
        }

        Tensor Tensor::get_grad() const
        {
            return this->tensor_data.get()->get_grad();
        }

        Tensor::Iterator Tensor::begin() const
        {
            return Iterator(*this, 0);
        }

        Tensor::Iterator Tensor::end() const
        {
            return Iterator(*this, this->get_buffer().shape().begin()[0]);
        }

        Tensor::Iterator Tensor::cbegin() const
        {
            return Iterator(*this, 0);
        }

        Tensor::Iterator Tensor::cend() const
        {
            return Iterator(*this, this->get_buffer().shape().begin()[0]);
        }

        const TensorBase& Tensor::get_buffer() const
        {
            return static_cast<bool>(this->tensor_data) ? this->tensor_data->buf : throw 0;
        }

        Tensor derive_reshape_cast(const Tensor& dat, const Tensor& new_shape, bool, const DataBuffer&)
        {
            return dat.reshape(new_shape.get_buffer().shape(), false).tensor_cast(new_shape.get_buffer().type(), false);
        }

        Tensor Tensor::reshape(const std::initializer_list<unsigned int>& list_dim, bool is_derive) const
        {
            assert(this->get_buffer().data_size() / get_sizeof_type(this->get_buffer().type()) == calculate_size(list_dim));
            std::vector<std::pair<Tensor, Derivation>> temp;
            if (is_derive)
                temp.push_back(std::make_pair(*this, Derivation(*this, derive_reshape_cast)));
            TensorBase other_buf(this->get_buffer().type(), list_dim, this->get_buffer().data());
            return Tensor(other_buf, temp);
        }

        Tensor Tensor::tensor_cast(const std::type_info& dtype) const
        {
            return tensor_cast(dtype, true);
        }

        Tensor Tensor::expand(unsigned char dim, unsigned int size)
        {
            std::vector<unsigned int> resize = this->get_buffer().shape();
            assert(dim < resize.size());
            unsigned int dim_x = 1;
            for (unsigned int i = 0; i < dim; i++)
                dim_x *= resize[i];
            std::vector<Tensor> arr;
            resize.erase(resize.begin(), std::next(resize.begin(), dim));
            resize.insert(resize.begin(), dim_x);
            Tensor temp_tensor = this->reshape(resize);
            for (Tensor item: temp_tensor)
            {
                std::vector<Tensor> temp_vec_t(size);
                for (Tensor& t: temp_vec_t)
                    t = item;
                arr.push_back(add_dim(temp_vec_t));
            }
            resize = this->get_buffer().shape();
            resize[dim] = size;
            return add_dim(arr).reshape(resize);
        }

        Tensor Tensor::reshape(const std::initializer_list<unsigned int>& dim_sizes) const
        {
            return this->reshape(dim_sizes, true);
        }

        Tensor Tensor::reshape(const std::vector<unsigned int>& dim_sizes) const
        {
            return this->reshape(wrapper::initializer_wrapper(dim_sizes.begin().operator->(), dim_sizes.end().operator->()));
        }

        Tensor derive_index(const Tensor& in_value, const Tensor&, bool, const DataBuffer& dat)
        {
            const unsigned int(*index1)[2] = static_cast<const unsigned int(*)[2]>(dat.get_data());
            std::vector<unsigned int> temp_dim = in_value.get_buffer().shape();
            temp_dim.insert(temp_dim.begin(), (*index1)[0]);
            Tensor tensor_temp = zeros<int>(temp_dim);
            void* temp_ptr = reinterpret_cast<void*>(reinterpret_cast<unsigned long long int>(tensor_temp.get_buffer().data()) + (*index1)[1] * in_value.get_buffer().data_size());
            devices::device_memcpy(temp_ptr, tensor_temp.get_buffer().get_device(), in_value.get_buffer().data(), in_value.get_buffer().get_device(), in_value.get_buffer().data_size());
            return tensor_temp;
        }

        Tensor Tensor::operator[](unsigned int index) const
        {
            unsigned int temp_arr[2] {this->get_buffer().shape().begin()[0],index};
            return Tensor(this->get_buffer()[index], { std::make_pair(std::forward<const Tensor>(*this), Derivation(Tensor(), derive_index, false, temp_arr)) });
        }

        Tensor Tensor::operator[](const std::initializer_list<Slice>& slice_arr) const
        {
            return this->slice(slice_arr, true);
        }

        Tensor Tensor::operator+() const
        {
            return multiply(*this, values(this->get_buffer().shape(), 1));
        }

        Tensor operator+(const Tensor& a, const Tensor& b)
        {
            return add(a, b);
        }

        Tensor Tensor::operator-() const
        {
            return multiply(*this, values(this->get_buffer().shape(), -1.f));
        }

        Tensor& Tensor::operator+=(const Tensor& other)
        {
            return this->operator=((*this) + other);
        }

        Tensor& Tensor::operator-=(const Tensor& other)
        {
            return this->operator=((*this) - other);
        }

        Tensor& Tensor::operator*=(const Tensor& other)
        {
            return this->operator=((*this) * other);
        }

        Tensor& Tensor::operator/=(const Tensor& other)
        {
            return this->operator=((*this) / other);
        }

        Tensor operator-(const Tensor& a, const Tensor& b)
        {
            return add(a, -b);
        }

        Tensor operator*(const Tensor& a, const Tensor& b)
        {
            return multiply(a, b);
        }

        Tensor operator/(const Tensor& a, const Tensor& b)
        {
            return divide(a, b);
        }

        Tensor operator!=(const Tensor& a, const Tensor& b)
        {
            return a<b || a>b;
        }

        Tensor operator==(const Tensor& a, const Tensor& b)
        {
            return !(a != b);
        }

        Tensor operator>=(const Tensor& a, const Tensor& b)
        {
            return !(a < b);
        }

        Tensor operator<=(const Tensor& a, const Tensor& b)
        {
            return !(a > b);
        }

        Tensor Tensor::exp() const
        {
            return this->exp(true);
        }

        Tensor Tensor::sin() const
        {
            return this->sin(true);
        }

        Tensor Tensor::cos() const
        {
            return this->cos(true);
        }

        Tensor Tensor::tan() const
        {
            return this->tan(true);
        }

        Tensor Tensor::sinh() const
        {
            return this->sinh(true);
        }

        Tensor Tensor::cosh() const
        {
            return this->cosh(true);
        }

        Tensor Tensor::tanh() const
        {
            return this->tanh(true);
        }

        Tensor Tensor::sigmoid() const
        {
            return this->sigmoid(true);
        }

        Tensor tensor_file_load(const char* dir)
        {
            if (std::FILE* tensor_file = std::fopen(dir, "rb"))
            {
                datatype::DataType type_wrapper;
                size_t temp_read;
                temp_read = std::fread(&type_wrapper, sizeof(datatype::DataType), 1, tensor_file);
                unsigned char shape_size;
                temp_read = std::fread(&shape_size, sizeof(char), 1, tensor_file);
                unsigned int* temp_shape = new unsigned int[shape_size];
                temp_read = std::fread(temp_shape, sizeof(unsigned int), shape_size, tensor_file);
                std::size_t total_dim_size = 1;
                for (unsigned char i = 0; i < shape_size; i++)
                    total_dim_size *= temp_shape[i];
                const std::type_info& dtype = datatype::warp_type(type_wrapper);
                void* temp_data = operator new(total_dim_size * get_sizeof_type(dtype));
                temp_read = std::fread(temp_data, 1, get_sizeof_type(dtype) * total_dim_size, tensor_file);
                std::fclose(tensor_file);
                TensorBase t_base(dtype, wrapper::initializer_wrapper(temp_shape, temp_shape + shape_size), temp_data, devices::DEVICE_CPU_0);
                operator delete(temp_data);
                delete[] temp_shape;
                return t_base;
            }
            return Tensor();
        }

        std::ostream& operator<<(std::ostream& out_stream, const Tensor& tensor_out)
        {
            std::string enter;
            if (tensor_out.get_buffer().shape().size())
            {
                out_stream << '{';
                for (Tensor it : tensor_out)
                {
                    out_stream << enter << it;
                    if (enter.size() == 0)
                    {
                        if (tensor_out.get_buffer().shape().size() == 1)
                            enter = ", ";
                        else
                            enter = ", \n";
                    }
                }
                out_stream << '}';
            }
            else
            {
#define ADD_CODE(TEMP)\
if (tensor_out.get_buffer().type() == typeid(TEMP))\
out_stream << static_cast<TEMP>(tensor_out);
                LOOP((bool)USING_DATA_TYPE);
#undef ADD_CODE
            }
            return out_stream;
        }

        Tensor power(const Tensor& a, const Tensor& b)
        {
            std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, b);
            return power(broadcast_t.first, broadcast_t.second, true);
        }

        Tensor add(const Tensor& a, const Tensor& b)
        {
            std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, b);
            return add(broadcast_t.first, broadcast_t.second, true);
        }

        Tensor multiply(const Tensor& a, const Tensor& b)
        {
            std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, b);
            return multiply(broadcast_t.first, broadcast_t.second, true, nullptr);
        }
        Tensor divide(const Tensor& a, const Tensor& b)
        {
            std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, b);
            return divide(broadcast_t.first, broadcast_t.second, true);
        }
        Tensor dot(const Tensor& a, const Tensor& b)
        {
            std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, b);
            return dot(broadcast_t.first, broadcast_t.second, true, nullptr);
        }

        Tensor matmul(const Tensor& a, const Tensor& b)
        {
            std::initializer_list<unsigned int> shape_a = a.get_buffer().shape();
            std::initializer_list<unsigned int> shape_b = b.get_buffer().shape();
            if ((shape_a.size() == 1 && shape_b.size() == 1) || shape_a.size() == 0 || shape_b.size() == 0)
                return dot(a, b);
            if (shape_a.size() == 1 && shape_b.size() == 2)
            {
                Tensor temp_a = a.reshape({ 1, shape_a.begin()[0] });
                return matmul(temp_a, b, true, nullptr).reshape({ shape_b.end()[-1] });
            }
            if (shape_a.size() == 2 && shape_b.size() == 1)
            {
                Tensor temp_b = a.reshape({ shape_b.begin()[0], 1 });
                return matmul(a, temp_b, true, nullptr).reshape({ shape_a.begin()[0] });
            }
            if (shape_a.size() == 2 && shape_b.size() == 2)
                return matmul(a, b, true, nullptr);
            if (shape_a.size() > 2 || shape_b.size() > 2)
            {
                Tensor temp_b = b;
                if (shape_a.size() == 1)
                    temp_b = temp_b.reshape({ shape_b.begin()[0], 1 });
                std::pair<Tensor, Tensor> broadcast_t = tensor_broadcasting(a, temp_b, 0, 2);
                return batchedmatmul(broadcast_t.first, broadcast_t.second, true, nullptr);
            }
        }

        Tensor condition(const Tensor& value_bool, const Tensor& value_true, const Tensor& value_false)
        {
            return condition(value_bool, value_true, value_false, true);
        }

        Tensor tensor_rand(const std::vector<unsigned int>& shape_vector, unsigned int seed)
        {
            return tensor_rand(wrapper::initializer_wrapper(shape_vector.begin().operator->(), shape_vector.end().operator->()), seed);
        }

        Tensor::Iterator::Iterator(reference_left ref, unsigned int index):
            ref(ref),
            index(index)
        {
        }
        Tensor::Iterator::reference Tensor::Iterator::operator*() const
        {
            return this->ref[index];
        }
        Tensor::Iterator& Tensor::Iterator::operator++()
        {
            index++;
            return *this;
        }
        Tensor::Iterator& Tensor::Iterator::operator--()
        {
            index--;
            return *this;
        }
        Tensor::Iterator Tensor::Iterator::operator++(int)
        {
            Iterator temp(*this);
            index++;
            return temp;
        }

        Tensor::Iterator Tensor::Iterator::operator--(int)
        {
            Iterator temp(*this);
            index--;
            return temp;
        }

        bool typeinfo_is_floating_point(const std::type_info& dtype)
        {
            return dtype == typeid(float) || dtype == typeid(double);
        }

        const std::type_info& comparison_type(const std::type_info& a, const std::type_info& b)
        {
            if (typeinfo_is_floating_point(a) == typeinfo_is_floating_point(b))
                return (get_sizeof_type(a) > get_sizeof_type(b)) ? a : b;
            if (typeinfo_is_floating_point(a))
                return a;
            if (typeinfo_is_floating_point(b))
                return b;
            throw std::exception();
        }

        TensorContentDerivation::TensorContentDerivation(const TensorBase& buf, const std::vector<std::pair<Tensor, Derivation>>& derive_data):
            TensorContent(buf),
            derive_data(check_derive_data(derive_data))
        {
        }

        TensorContentDerivation::TensorContentDerivation(TensorBase&& buf, std::vector<std::pair<Tensor, Derivation>>&& derive_data) :
            TensorContent(std::forward<TensorBase>(buf)),
            derive_data(check_derive_data(derive_data))
        {
        }

        void* create_mem_101(std::size_t s, const void* dat)
        {
            void* temp_sizes = operator new(s);
            std::memcpy(temp_sizes, dat, s);
            return temp_sizes;
        }

        bool operator==(const DataBuffer& a, const DataBuffer& b)
        {
            return a.data_size == b.data_size && std::memcmp(a.data, b.data, std::min(a.data_size, b.data_size)) == 0;
        }

        DataBuffer::DataBuffer():
            data(nullptr),
            data_size(0)
        {
        }

        DataBuffer::DataBuffer(std::nullptr_t null):
            data(null),
            data_size(0)
        {
        }

        DataBuffer::DataBuffer(const DataBuffer& other):
            data(create_mem_101(other.data_size, other.data)),
            data_size(other.data_size)
        {
        }
        DataBuffer::~DataBuffer()
        {
            delete this->data;
        }
        const void* const& DataBuffer::get_data() const
        {
            return this->data;
        }
        const std::size_t& DataBuffer::get_data_size() const
        {
            return this->data_size;
        }
        DataBuffer& DataBuffer::operator=(const DataBuffer& other)
        {
            delete data;
            data = create_mem_101(other.data_size, other.data);
            data_size = other.data_size;
            return *this;
        }
        Tensor::TensorContent::TensorContent(const TensorBase& buf):
            buf(buf)
        {
        }
        Tensor::TensorContent::TensorContent(TensorBase&& buf):
            buf(std::forward<TensorBase>(buf))
        {
        }

        WeakTensor::WeakTensor(const Tensor& other):
            tensor_data(other.tensor_data)
        {
        }
        bool WeakTensor::is_tensor_expired()
        {
            return this->tensor_data.expired();
        }
        Tensor WeakTensor::to_tensor()
        {
            return this->tensor_data.lock();
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

#undef USING_DATA_TYPE
