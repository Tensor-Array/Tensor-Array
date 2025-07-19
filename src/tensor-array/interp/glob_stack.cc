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

