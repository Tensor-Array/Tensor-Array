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
#include "parser.h"
#include "token.h"
#include "open_file.h"
#include "vm_type.h"

void emit(int size, ...)
{
    va_list args;
    va_start(args, size);
    
    // Process the variable arguments as needed
    for (int i = 0; i < size; ++i) {
        ++text;
        *text = va_arg(args, VM_INSTRUCTION);
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

void expression(int level)
{
    sym_data* temp = NULL; // Temporary variable to hold intermediate values
    int isArrRef = 0; // Flag to check if we are dealing with an array reference
    // This function would handle parsing and evaluating expressions
    // For now, it is a placeholder
    // You can implement your logic here
    switch (tkn)
    {
    case TOKEN_NUM:
        /* code */
        emit(3, IMM, TYPE_INT, tkn_val);
        match(TOKEN_NUM);
        break;
    case TOKEN_ID:
        /* code */
        temp = sym_cur;
        match(TOKEN_ID);
        if (temp->type)
        {
            if (token == '(')
            {
                /* code */
                match('(');
                match(')');
                emit(2, CALL, temp->data)
            }
            
        }
        else
        {
            emit(3, IMM, TYPE_PTR, tkn_val);
            emit(1, GET);
        }
        break;
    case '"':
        {
            emit(3, IMM, TYPE_STRING, tkn_val);
            match('"'); // Match the opening quote
        }
        break;
    case '[':
        if (temp == NULL)
        {
            *text = PTR_PUSH; // Push the current value onto the stack
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
            *text = PTR_PUSH; // Push the current value onto the stack
            match(TOKEN_ASSIGN);
            expression(TOKEN_ASSIGN); // Parse the right-hand side expression
            if (isArrRef) emit(1, SETELEM); // Emit set element instruction if it's an array reference
            else emit(1, SET); // Emit set instruction
            break;
        case TOKEN_EQ:
            emit(1, PUSH);
            match(TOKEN_EQ);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, EQ); // Emit equality instruction
            break;
        case TOKEN_NE:
            emit(1, PUSH);
            match(TOKEN_NE);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, NE); // Emit not equal instruction
            break;
        case TOKEN_LT:
            emit(1, PUSH);
            match(TOKEN_LT);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, LT); // Emit less than instruction
            break;
        case TOKEN_GT:
            emit(1, PUSH);
            match(TOKEN_GT);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, GT); // Emit greater than instruction
            break;
        case TOKEN_LE:
            emit(1, PUSH);
            match(TOKEN_LE);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, LE); // Emit less than or equal instruction
            break;
        case TOKEN_GE:
            emit(1, PUSH);
            match(TOKEN_GE);
            expression(TOKEN_SHL); // Parse the right-hand side expression
            emit(1, GE); // Emit greater than or equal instruction
            break;
        case TOKEN_SHL:
            emit(1, PUSH);
            match(TOKEN_SHL);
            expression(TOKEN_ADD); // Parse the right-hand side expression
            emit(1, SHL); // Emit shift left instruction
            break;
        case TOKEN_SHR:
            emit(1, PUSH);
            match(TOKEN_SHR);
            expression(TOKEN_ADD); // Parse the right-hand side expression
            emit(1, SHR); // Emit shift right instruction
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
        case TOKEN_MUL:
            emit(1, PUSH);
            match(TOKEN_MUL);
            expression(TOKEN_MATMUL); // Parse the right-hand side expression
            emit(1, MUL); // Emit multiply instruction
            break;
        case TOKEN_DIV:
            emit(1, PUSH);
            match(TOKEN_DIV);
            expression(TOKEN_MATMUL); // Parse the right-hand side expression
            emit(1, DIV); // Emit divide instruction
            break;
        case TOKEN_MATMUL:
            emit(1, PUSH);
            match(TOKEN_MATMUL);
            expression(TOKEN_INC); // Parse the right-hand side expression
            emit(1, MATMUL); // Emit matrix multiply instruction
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
            VM_INSTRUCTION *b = ++text; // Placeholder for jump address
            statement(); // Parse the statement inside the if block
            if (tkn == TOKEN_ELSE)
            {
                match(TOKEN_ELSE);
                emit(1, JMP); // Emit jump instruction
                *b = text + 2; // Set the jump address to the next instruction
                b = ++text;
                statement(); // Parse the else block
            }
            *b = text + 1; // Set the jump address to the next instruction
        }
        break;
    case TOKEN_WHILE:
        {
            VM_INSTRUCTION *a = NULL; // Placeholder for jump address
            VM_INSTRUCTION *b = text+1; // Placeholder for jump address
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
    case TOKEN_FUNC:
        match(TOKEN_FUNC);
        if (tkn != TOKEN_ID)
        {
            fprintf(stderr, "Error: function name\n");
            exit(1);
        }
        cur->type = TYPE_FUNC;
        cur->data = malloc(1024*8);
        VM_INSTRUCTION *save  = text;
        text = cur->data
        match(TOKEN_ID);
        match('(');
        match(')');
        statement();
        if (*text != RET) emit(1, RET);
        text = save;
        break;
    case: TOKEN_RETURN:
        match(TOKEN_RETURN);
        expression(TOKEN_ASSIGN);
        emit(1, RET);
        break;
    case '{':
        match('{');
        while (tkn != '}')
            statement();
        match('}');
        break;
    case '\0':
        return;
    default:
        expression(TOKEN_ASSIGN);
        if (tkn == ';')
            match(';');
        break;
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
        orig = text+1;
        char *isrc = src;
        VM_INSTRUCTION *itext = text;
        interp_memreset();
        printf(">>> ");
        fflush(stdout);
        fgets(src, poolsize-1, stdin); // Read input from stdin
        token_next();
        statement();
        emit(1, EXIT); // Emit a token with value 0 to indicate end of processing
        eval();
        printf("eval \n");
        puts("");
        free(itext);
        free(isrc);
    }
}
