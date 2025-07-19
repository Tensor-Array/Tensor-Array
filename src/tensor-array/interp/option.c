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

#include <stdio.h>

void help()
{
    printf("Usage: tensor-array [options]\n");
    printf("Options:\n");
    printf("  -h, --help                  Show this help message\n");
    printf("  -v, --version               Show version information\n");
    printf("      --poolsize    SIZE      Set the pool size (default: 1024)\n");
    printf("  -f, --file        FILE      Open the specified file\n");
}

void version()
{
    printf("Tensor Array Interpreter Version 0.1.0\n");
}