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
#include <stdlib.h>
#include <string.h>

char *src, *text = NULL;
size_t poolsize = 1024; // Default pool size

void interp_malloc()
{
    src = malloc(poolsize);
    if (src == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for interpreter\n");
        exit(1);
    }
    text = malloc(poolsize);
    if (text == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for interpreter text\n");
        free(src);
        exit(1);
    }
}

void interp_memreset()
{
    memset(text, 0, poolsize);
    memset(src, 0, poolsize);
}

void interp_free()
{
    free(text);
    free(src);
}

void read_file(const char* filename)
{
    FILE* fptr = fopen(filename, "r");
    if (fptr == NULL)
    {
        fprintf(stderr, "Error: Could not open file %s\n", filename);
        exit(1);
    }
    
    int i;
    interp_malloc();
    i = fread(src, poolsize, 1, fptr);
    if (i < 0)
    {
        fprintf(stderr, "Error: Could not read file %s\n", filename);
        fclose(fptr);
        exit(1);
    }
    return 0; // Return 0 on success
}
