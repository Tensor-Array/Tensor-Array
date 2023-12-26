// TensorTesting.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

/*
#include "linear.hh"
#include "normalization.hh"
#include "recurrent.hh"
#include "convolution.hh"
#include <iostream>
#define NAME(x) #x
using namespace tensor_array::layers;
using namespace tensor_array::value;

class BeginModelImpl : public TensorCalculateLayerImpl
{
private:
    Linear a1 = 64;
    Normalization a2 = std::initializer_list<unsigned char>{ 0 };
    LSTM a4 = 64;
public:
    Tensor calculate(const Tensor&) override;
    BeginModelImpl();
};

using BeginModel = LayerHolder<BeginModelImpl>;

int main(int argc, char* argv[])
{
    tensor_array::devices::device_CUDA_get_info();
    Tensor ones1001 = tensor_array::layers::tanh(tensor_array::value::tensor_rand({ 8, 1, 1, }));
    Conv1D conv(dimension{ 1 }, 1);
    auto test0010 = conv(ones1001);
    std::cout << "test conv2d = " << test0010 << std::endl;


    std::time_t time_begin = std::time(0);
    decltype(1L + 1U) a0001;
    float a3[2][3] =
    {
        {1, 2, 3},
        {4, 5, 6},
    };
    float a4[3][2] =
    {
        {1, 2},
        {3, 4},
        {5, 6}
    };
    sizeof tensor_array::value::TensorBase;
    tensor_array::value::TensorArray<float, 2U, 3U> a001{ 1, 2, 3, 4, 5, 6 };
    tensor_array::value::TensorArray<float, 3U, 2U> a002{ 1, 2, 3, 4, 5, 6 };
    sizeof a001;
    //tensor_array::value::TensorArray<int> a003 = { 1 };
    std::array<std::array<int, 3>, 3> a123{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    tensor_array::value::TensorBase a(a001);
    tensor_array::value::TensorBase b(a002);
    Tensor a1(a);
    Tensor b1(b);
    std::printf("a1 = %f\n", float(a1[0][0]));
    std::printf("a1 type name: %s\n", a1.get_buffer().type().name());
    Tensor c1 = matmul(a1, b1);
    std::printf("%f\n", float(c1[0][0]));
    Tensor c1_ones = values(c1.get_buffer().shape(), 1.f);
    Tensor d1 = power(c1, c1_ones);
    d1.calc_grad();
    std::printf("tuple = %li;\n", a1.tensor_use_count());

    //Tensor d1 = c1.transpose(0, 1);
    //std::printf("%f\n", float(d1[0]));
    //std::printf("%f\n", float(b1.get_grad()[0]));

    std::cout << "d1 = " << d1 << std::endl;

    std::cout << "c1 = " << c1.mean(0) << std::endl;
    std::cout << "a1 grad = " << a1.get_grad() << std::endl;

    //tensor_array.random_weights();
    std::printf("%s\n", NAME(1 + 1 = 3));

    // My new machine learning. (Tensor structure)
    // Look like Python neural network libraries (TensorFlow, Keras, PyTorch,...) but its C++.
    // C++ is faster than Python.
    Tensor ones = tensor_array::value::tensor_rand({ 8, 8, 73 });
    ones.save("test/nothing.dat");
    ones = tensor_array::value::tensor_file_load("begin.dat");
    ones.save("begin.dat");
    ones = tensor_array::value::tensor_file_load("begin.dat");
    //ones.save("begin.dat");
    ones = ones.reshape({ 1, 4672 });
    Tensor ones1 = tensor_array::value::tensor_rand({ 64 });
    std::printf("ones = %f\n", float(ones[0][0]));
    BeginModel model_1;
    model_1->load_data("test1/test2");
    auto test055 = model_1(ones);
    Tensor n1 = tensor_array::value::add_dim({ model_1(ones), model_1(ones), model_1(ones), });
    n1.multithread_derive() = true;
    Tensor n2 = n1.loss(ones1);
    n2.calc_grad();
    model_1->update_weight(0.1);
    model_1->save_data("test1/test2");
    std::cout << "n1 loss = " << n2 << std::endl;
    std::cout << "n1 = " << std::endl << n1 << std::endl;

    std::printf("\n");
    std::time_t time_end = std::time(0);
    std::printf("time completed: %llu s\n", time_end - time_begin);
    return 0;
}

BeginModelImpl::BeginModelImpl()
{
    this->map_layer.insert(std::make_pair("Dense", a1.get_shared()));
    this->map_layer.insert(std::make_pair("BatchNorm", a2.get_shared()));
    this->map_layer.insert(std::make_pair("LSTM", a4.get_shared()));
    a4->set_copy_after_calculate(true);
}

Tensor BeginModelImpl::calculate(const Tensor& in)
{
    Tensor temp = in;
    temp = a1(temp);
    temp = a2(temp);
    temp = ReLU(temp);
    temp = a4(temp);
    temp = temp.reshape({ 64 });
    return temp;
}
*/
// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
