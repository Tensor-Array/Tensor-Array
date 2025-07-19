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
    void op_add();
    void op_sub();
    void op_mul();
    void op_div();
    void op_matmul();
    void op_pos();
    void op_neg();
    void op_and();
    void op_or();
    void op_not();
    void op_eq();
    void op_ne();
    void op_lt();
    void op_gt();
    void op_le();
    void op_ge();
    void op_shl();
    void op_shr();
    void op_open();
    void op_read();
    void op_close();
    void op_prtf();
    void op_malc();
    void op_mset();
    void op_mcmp();
    void op_exit();
    void op_push();
    void op_get();
    void op_set();
#ifdef __cplusplus
}
#endif

