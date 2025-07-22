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
#include <cstring>
#include "sym_map.h"
#include "vmop.h"
#include "vm_type.h"

typedef long VM_INSTRUCTION;
extern VM_INSTRUCTION* pc;

std::stack<tensor_array::value::Tensor> tensor_stack;
std::stack<std::string> var_stack;
std::stack<std::pair<VM_INSTRUCTION*, scope>> call_stack;
tensor_array::value::Tensor ag;
void* aptr;
long any_value;
long any_type;

void new_int()
{
    tensor_array::value::TensorArray<int> tmp2 = {any_value};
    tensor_array::value::Tensor tmp1(tmp2);
    ag = tmp1;
}

void new_ptr()
{
    aptr = reinterpret_cast<void*>(any_value);
}

void new_string()
{
    char* str = reinterpret_cast<char*>(any_value);
    unsigned int s_len = std::strlen(str);
    tensor_array::value::TensorBase tmp1(typeid(char),{s_len}, str);
    ag = tmp1;
    std::free(str);
}

void op_imm()
{
    switch (any_type)
    {
    case TYPE_INT:
        /* code */
        new_int();
        break;
    case TYPE_STRING:
        new_string();
        break;
    case TYPE_PTR:
        new_ptr();
        break;
    default:
        break;
    }
}

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

void op_call()
{
    VM_INSTRUCTION* pc1 = (VM_INSTRUCTION*)*pc++;
    call_stack.push({std::move(pc), std::move(sym_map)});
    pc = pc1;
}

void op_ret()
{
    pc = call_stack.top().first;
    sym_map = std::move(call_stack.top().second);
    call_stack.pop();
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
    // std::cout << ag << std::endl;
}

void op_push()
{
    tensor_stack.push(ag);
}

void op_ptr_push()
{
    var_stack.push(reinterpret_cast<char*>(aptr));
    std::free(aptr);
}

void op_get()
{
    char *var_name = reinterpret_cast<char*>(aptr);
    sym_data& temp = sym_map[var_name];
    std::free(aptr);
    ag = *reinterpret_cast<tensor_array::value::Tensor*>(temp.data);
}

void op_set()
{
    if (!var_stack.empty())
    {
        std::string& var_name = var_stack.top();
        sym_data& temp = sym_map[var_name];
        delete temp.data; // Set the top of the stack to ag
        temp.data = new tensor_array::value::Tensor(ag);
        var_stack.pop();
    }
    else
    {
        throw std::runtime_error("Tensor stack is empty");
    }
}
