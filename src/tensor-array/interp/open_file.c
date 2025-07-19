
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
