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

#include <stack>
#include <tensor-array/core/tensor.hh>
#include <iostream>

std::stack<tensor_array::value::Tensor> tensor_stack;
tensor_array::value::Tensor ag;

void op_add()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag += tensor_stack.top();
    tensor_stack.pop();
}

void op_sub()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag -= tensor_stack.top();
    tensor_stack.pop();
}

void op_mul()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag *= tensor_stack.top();
    tensor_stack.pop();
}

void op_div()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag /= tensor_stack.top();
    tensor_stack.pop();
}

void op_matmul()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = tensor_array::value::matmul(ag, tensor_stack.top());
    tensor_stack.pop();
}

void op_pos()
{
    ag = +ag;
}

void op_neg()
{
    ag = -ag;
}

void op_and()
{
    ag = ag && tensor_stack.top();
    tensor_stack.pop();
}

void op_or()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag || tensor_stack.top();
    tensor_stack.pop();
}

void op_not()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = !ag;
}

void op_eq()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag == tensor_stack.top();
    tensor_stack.pop();
}

void op_ne()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag != tensor_stack.top();
    tensor_stack.pop();
}

void op_lt()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag < tensor_stack.top();
    tensor_stack.pop();
}

void op_gt()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag > tensor_stack.top();
    tensor_stack.pop();
}

void op_le()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag <= tensor_stack.top();
    tensor_stack.pop();
}

void op_ge()
{
    if (tensor_stack.empty())
    {
        throw std::runtime_error("Tensor stack is empty");
    }
    ag = ag >= tensor_stack.top();
    tensor_stack.pop();
}

void op_shl()
{
    // ag = ag << bg;
}

void op_shr()
{
    // ag = ag >> bg;
}

void op_open()
{
    // Implementation for opening a file or resource
}

void op_read()
{
    // Implementation for reading from a file or resource
}

void op_close()
{
    // Implementation for closing a file or resource
}

void op_prtf()
{
    // Implementation for printing formatted output
}

void op_malc()
{
    // Implementation for memory allocation
}

void op_mset()
{
    // Implementation for setting memory
}

void op_mcmp()
{
    // Implementation for memory comparison
}

void op_exit()
{
    // Implementation for exiting the program
    std::cout<< ag << std::endl;
}

void op_push()
{
    tensor_stack.push(ag);
}

void op_get()
{
    if (!tensor_stack.empty())
    {
        ag = tensor_stack.top();
        tensor_stack.pop();
    }
    else
    {
        throw std::runtime_error("Tensor stack is empty");
    }
}

void op_set()
{
    if (!tensor_stack.empty())
    {
        tensor_array::value::Tensor bg = tensor_stack.top();
        tensor_stack.pop();
        ag = bg; // Set the top of the stack to ag
    }
    else
    {
        throw std::runtime_error("Tensor stack is empty");
    }
}
