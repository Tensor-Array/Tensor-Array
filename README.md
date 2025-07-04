# Tensor-Array

![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
[![Docker Image Size with architecture (latest by date/latest semver)](https://img.shields.io/docker/image-size/noobwastaken/tensor-array)
](https://hub.docker.com/repository/docker/noobwastaken/tensor-array/general)


A C++ Tensor library that can be used to work with machine learning or deep learning project.

Build your own neural network models with this library.



## Why this repository named `Tensor-Array`

We created a template struct that named `TensorArray`. That struct is a multi-dimensional array wrapper.

```C++
#include "tensor_array/core/tensorbase.hh"

using namespace tensor_array::value;

int main()
{
  TensorArray<float, 4, 4> example_tensor_array =
  {{
    {{ 1, 2, 3, 4 }},
    {{ 5, 6, 7, 8 }},
    {{ 9, 10, 11, 12 }},
    {{ 13, 14, 15, 16 }},
  }};
  return 0;
}

```

That code is wrapping for:

```C++
int main()
{
  float example_tensor_array[4][4] =
  {
    { 1, 2, 3, 4 },
    { 5, 6, 7, 8 },
    { 9, 10, 11, 12 },
    { 13, 14, 15, 16 },
  };
  return 0;
}

```

## The `Tensor` class.

The `Tensor` class is a storage that store value and calculate the tensor.

The `Tensor::calc_grad()` method can do automatic differentiation.

The `Tensor::get_grad()` method can get the gradient after call `Tensor::calc_grad()`.


```C++
#include <iostream>
#include "tensor_array/core/tensor.hh"

using namespace std;
using namespace tensor_array::value;

int main()
{
  TensorArray<float, 4, 4> example_tensor_array =
  {{
    {{ 1, 2, 3, 4 }},
    {{ 5, 6, 7, 8 }},
    {{ 9, 10, 11, 12 }},
    {{ 13, 14, 15, 16 }},
  }};
  TensorArray<float> example_tensor_array_scalar = {100};
  Tensor example_tensor_1(example_tensor_array);
  Tensor example_tensor_2(example_tensor_array_scalar);
  Tensor example_tensor_sum = example_tensor_1 + example_tensor_2;
  cout << example_tensor_sum << endl;
  example_tensor_sum.calc_grad();
  cout << example_tensor_1.get_grad() << endl;
  cout << example_tensor_2.get_grad() << endl;
  return 0;
}

```
