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

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include "open_file.h"
#include "parser.h"
#include "token.h"
#include "vm.h"

void emit(int size, ...)
{
    va_list args;
    va_start(args, size);
    
    // Process the variable arguments as needed
    for (int i = 0; i < size; ++i) {
        ++text;
        *text = va_arg(args, long);
    }
    
    va_end(args);
}

void match(long tk)
{
    if (tkn == tk) {
        token_next(); // Move to the next token
    } else {
        if (tk < 0x80) {
            fprintf(stderr, "Error: Expected token %ld but found %ld\n", tk, tkn);
        } else {
            char* tn = tknname[tk - 0x80];
            fprintf(stderr, "Error: Expected token %s but found %s\n", tn, tkn);
        }
        exit(1);
    }
}

void program()
{
    while (1)
    {
        // This is a placeholder for the main program loop
        // You would typically call emit or other functions here based on your program logic
        // Add your program logic here
        interp_malloc();
        char *isrc = src;
        char *itext = text;
        interp_memreset();
        printf(">>> ");
        read(0, src, poolsize-1); // Read input from stdin
        token_next();
        statement();
        emit(1, EXIT); // Emit a token with value 0 to indicate end of processing
        eval();
        puts("");
        free(itext);
        free(isrc);
    }
}

void expression(int level)
{
    void* temp = NULL; // Temporary variable to hold intermediate values
    int isArrRef = 0; // Flag to check if we are dealing with an array reference
    // This function would handle parsing and evaluating expressions
    // For now, it is a placeholder
    // You can implement your logic here
    switch (tkn)
    {
    case TOKEN_NUM:
        /* code */
        emit(1, IMM);
        match(TOKEN_NUM);
        break;
    case TOKEN_ID:
        /* code */
        emit(1, GET);
        match(TOKEN_ID);
        break;
    case '"':
        {
            match('"'); // Match the opening quote
        }
        break;
    case '[':
        if (temp == NULL)
        {
            *text = PUSH; // Push the current value onto the stack
            match('['); // Match the opening bracket
            expression(TOKEN_ASSIGN); // Parse the expression inside the brackets
            emit(1, GETELEM); // Emit get element instruction
            match(']'); // Match the closing bracket
        }
        break;
    default:
        break;
    }

    while (tkn >= level)
    {
        switch (tkn)
        {
        case TOKEN_ASSIGN:
            if (*text != GET && *text != GETELEM)
            {
                fprintf(stderr, "Error: Assignment without a variable\n");
                exit(1);
            }
            *text = PUSH; // Push the current value onto the stack
            match(TOKEN_ASSIGN);
            expression(TOKEN_ASSIGN); // Parse the right-hand side expression
            if (isArrRef) emit(1, SETELEM); // Emit set element instruction if it's an array reference
            else emit(1, SET); // Emit set instruction
            break;
        case TOKEN_ADD:
            emit(1, PUSH);
            match(TOKEN_ADD);
            expression(TOKEN_MUL); // Parse the right-hand side expression
            emit(1, ADD); // Emit add instruction
            break;
        case TOKEN_SUB:
            emit(1, PUSH);
            match(TOKEN_SUB);
            expression(TOKEN_MUL); // Parse the right-hand side expression
            emit(1, SUB); // Emit subtract instruction
            break;
        default:
            fprintf(stderr, "Error: Unrecognized token in expression\n");
            exit(1);
        }
    }
    
}

void statement()
{
    // This function would handle parsing and executing statements
    // For now, it is a placeholder
    // You can implement your logic here
    switch (tkn)
    {
    case TOKEN_IF:
        {
            match(TOKEN_IF);
            match('(');
            expression(TOKEN_ASSIGN); // Parse the condition expression
            match(')');
            emit(1, JZ); // Emit jump if zero instruction
            long *b = ++text; // Placeholder for jump address
            statement(); // Parse the statement inside the if block
            if (tkn == TOKEN_ELSE)
            {
                match(TOKEN_ELSE);
                emit(1, JMP); // Emit jump instruction
                *b = text + 2; // Set the jump address to the next instruction
                statement(); // Parse the else block
            }
            *b = text + 1; // Set the jump address to the next instruction
        }
        break;
    case TOKEN_WHILE:
        {
            long *a = NULL; // Placeholder for jump address
            long *b = text+1; // Placeholder for jump address
            match(TOKEN_WHILE);
            match('(');
            expression(TOKEN_ASSIGN); // Parse the condition expression
            match(')');
            emit(1, JZ); // Emit jump if zero instruction
            a=++text; // Set the jump address to the start of the while block
            statement(); // Parse the statement inside the while block
            emit(1, JMP); // Emit jump instruction to loop back
            emit(1, b); // Emit the address to jump back to
            *a = text + 1; // Set the jump address to the next instruction
        }
        break;
    default:
        break;
    }
}
