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

typedef enum
{
    LEA, IMM, JMP, CALL, JZ, JNZ, ENT, ADJ, LEV, RET, LI, LC, SI, SC, SET, GET, PUSH, PTR_PUSH, GETELEM, SETELEM, ADDELEM,
    OR, XOR, AND, EQ, NE, LT, GT, LE, GE, ADD, SUB, MUL, DIV, MATMUL, POS, NEG, NOT, SHL, SHR,
    OPEN, READ, CLOSE, PRTF, MALC, MSET, MCMP, EXIT
} VM_INSTRUCTION_V2;

typedef size_t VM_INSTRUCTION;

void eval();

extern size_t any_value;
extern VM_INSTRUCTION* orig;
