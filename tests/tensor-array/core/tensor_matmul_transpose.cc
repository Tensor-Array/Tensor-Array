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

#include <tensor-array/core/tensor.hh>

using namespace tensor_array::value;

int tensor_matmul_transpose(int argc, char *argv[])
{
    TensorArray<float, 2, 3> example_tensor_array =
    {{
        {{ 1, 2, 3 }},
        {{ 4, 5, 6 }}
    }};
    TensorArray<float> example_tensor_array_scalar = {100};
    Tensor example_tensor_1(example_tensor_array_1);
    Tensor example_tensor_2 = example_tensor_1.transpose(0, 1);
    Tensor example_tensor_add = matmul(example_tensor_1, example_tensor_2);
    return 0;
}
