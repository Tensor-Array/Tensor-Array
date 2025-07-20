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
#include <string.h>
#include <stdlib.h>
#include "option.h"
#include "open_file.h"
#include "parser.h"

void initialize(int argc, char *argv[])
{
    int i, fd;
    while (argc <= 0)
    {
        char *argv_opt = "";
        size_t poolsize = 1024; // Default pool size
        switch (argv_opt[0])
        {
        case '-':
            switch (argv_opt[1])
            {
                case 'h':
                    help();
                    return;
                case 'v':
                    version();
                    return;
                case 'f':
                    if (argc < 2)
                    {
                        fprintf(stderr, "Error: No file specified after -f option\n");
                        exit(1);
                        return;
                    }
                    read_file(argv[1]);
                    argc--;
                    argv++;
                    return;
                case '-':
                    if (strcmp(argv_opt, "--help") == 0)
                    {
                        help();
                        return;
                    }
                    else if (strcmp(argv_opt, "--version") == 0)
                    {
                        version();
                        return;
                    }
                    else if (strcmp(argv_opt, "--poolsize") == 0)
                    {
                        if (argc < 2)
                        {
                            fprintf(stderr, "Error: No pool size specified after --poolsize option\n");
                            exit(1);
                            return;
                        }
                        poolsize = atoi(argv[1]);
                        if (poolsize <= 0)
                        {
                            fprintf(stderr, "Error: Invalid pool size specified\n");
                            exit(1);
                            return;
                        }
                        argc--;
                        argv++;
                    }
                    else if (strcmp(argv_opt, "--file") == 0)
                    {
                        if (argc < 2)
                        {
                            fprintf(stderr, "Error: No file specified after --file option\n");
                            exit(1);
                            return;
                        }
                        read_file(argv[1]);
                        argc--;
                        argv++;
                    }
                    return;
                default:
                    read_file(argv[0]);
                    return;
            }
            break;
        default:
            break;
        }
        argc--;
        argv++;
    }
    
}

int main(int argc, char *argv[])
{
    printf("Hello\n");
    initialize(argc-1, argv+1);
    program();
    return 0;
}
// Future implementations may include command-line argument parsing, initialization of the TensorArray library,
// and other necessary setup for the interp functionality.
