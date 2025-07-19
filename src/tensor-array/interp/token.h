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
    TOKEN_NUM = 0x80, TOKEN_SYS, TOKEN_GLO, TOKEN_LOC, TOKEN_ID,
    TOKEN_FUNC, TOKEN_ELSE, TOKEN_ENUM, TOKEN_IF, TOKEN_RETURN, TOKEN_SIZEOF,
    TOKEN_WHILE, TOKEN_ASSIGN, TOKEN_COND, TOKEN_LOR, TOKEN_LAN,
    TOKEN_OR, TOKEN_XOR, TOKEN_AND, TOKEN_SHL, TOKEN_SHR,
    TOKEN_EQ, TOKEN_NE, TOKEN_LT, TOKEN_GT, TOKEN_LE, TOKEN_GE,
    TOKEN_ADD, TOKEN_SUB, TOKEN_MUL, TOKEN_DIV, TOKEN_MATMUL, TOKEN_POS, TOKEN_NEG, TOKEN_NOT
} TOKEN_TYPE;

char* tknname[] = {
    "num", "sys", "glo", "loc", "id",
    "func", "else", "enum", "if", "return", "sizeof",
    "while", "assign", "cond", "lor", "lan",
    "or", "xor", "and",
    "eq", "ne", "lt", "gt", "le", "ge",
    "add", "sub", "mul", "div", "matmul", "pos", "neg", "not", "brak"
};

void token_next();
extern long tkn = 0;
