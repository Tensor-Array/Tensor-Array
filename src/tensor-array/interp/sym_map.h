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

#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct
    {
        long tkn;
        long hash;
        long cls;
        void* data; // Pointer to additional data if needed
    } sym_data;
    void sym_data_set(char* name, sym_data dat);
    sym_data* sym_data_get(char*);
    int glob_data_find(char* name);
    extern sym_data* sym_cur;
    void* new_Tensor();
#ifdef __cplusplus
}

extern std::map<std::string, sym_data> sym_map;
#endif
