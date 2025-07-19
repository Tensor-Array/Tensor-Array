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

#include <map>
#include <cstring>
#include <tensor-array/core/tensor.hh>
#include "glob_stack.h"

std::map<std::string, glob_data_t> data_map;


void glob_data_set(char* name, glob_data_t item)
{
    data_map[name] = item;
}

glob_data_t glob_data_get(char* name)
{
    return data_map[name];
}

int glob_data_find(char* name)
{
    return data_map.find(name) != data_map.end();
}

