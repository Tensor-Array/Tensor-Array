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