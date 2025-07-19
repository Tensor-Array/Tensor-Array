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

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include "glob_stack.h"
#include "open_file.h"
#include "token.h"

long tkn = 0;
long token_val = 0; // Variable to hold the value of the current token

void token_next()
{
    glob_data_t token_item;
    while ((tkn = *src++) != '\0')
    {
        switch (tkn)
        {
        case ' ':
        case '\t':
            /* code */
            break;
        case '#':
            /* code */
            while (*src != '\n' && *src != '\0') src++;
            break;
        case '"':
        case '\'':
            {
                char* last_pos = src - 1;
                while (*src != tkn && *src != '\0')
                {
                    src++;
                }
                token_val = last_pos; // Store the start of the string literal
            }
            return; // Exit after processing the string literal
        case '/':
            switch (src[0])
            {
            case '/':
                /* code */
                while (*src != '\n' && *src != '\0') src++;
                break;
            case '*':
                /* code */
                src++;
                while (*src != '\0' && !(src[0] == '*' && src[1] == '/')) src++;
                if (*src == '\0') {
                    fprintf(stderr, "Error: Unmatched comment block\n");
                    exit(1);
                }
                src += 2; // Skip past the closing */
                break;
            case '=':
                src++;
                tkn = TOKEN_DIV; // Store the token value
                return; // Exit after processing the division operator
            default:
                tkn = TOKEN_DIV; // Store the token value
                return; // Exit after processing the division operator
            }
            
        case '*':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_MUL; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_MUL; // Store the token value
                return; // Exit after processing the token
            }
        case '+':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_ADD; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_ADD; // Store the token value
                return; // Exit after processing the token
            }
        case '-':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_SUB; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_SUB; // Store the token value
                return; // Exit after processing the token
            }
        case '=':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_EQ; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_ASSIGN; // Store the token value
                return; // Exit after processing the token
            }
        case '!':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_NE; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_NOT; // Store the token value
                return; // Exit after processing the token
            }
        case '<':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_LE; // Store the token value
                return; // Exit after processing the token
            }
            else if (*src == '<')
            {
                src++;
                tkn = TOKEN_SHL; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_LT; // Store the token value
                return; // Exit after processing the token
            }
        case '>':
            if (*src == '=')
            {
                src++;
                tkn = TOKEN_GE; // Store the token value
                return; // Exit after processing the token
            }
            else if (*src == '>')
            {
                src++;
                tkn = TOKEN_SHR; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                tkn = TOKEN_GT; // Store the token value
                return; // Exit after processing the token
            }
        case '&':
            if (*src == '&')
            {
                src++;
                tkn = TOKEN_AND; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                fprintf(stderr, "Error: Unrecognized token '&'\n");
                exit(1);
            }
        case '|':
            if (*src == '|')
            {
                src++;
                tkn = TOKEN_LOR; // Store the token value
                return; // Exit after processing the token
            }
            else
            {
                fprintf(stderr, "Error: Unrecognized token '|'\n");
                exit(1);
            }
        case '@':
            src++;
            tkn = TOKEN_MATMUL; // Store the token value
            return; // Exit after processing the token
        default:
            if (tkn >= '0' && tkn <= '9')
            {
                if (tkn == '0' && (*src == 'x' || *src == 'X'))
                {
                    src++;
                    while ((*src >= '0' && *src <= '9') || (*src >= 'a' && *src <= 'f') || (*src >= 'A' && *src <= 'F'))
                    {
                        token_val = (token_val << 4) + (*src >= '0' && *src <= '9' ? *src - '0' : (*src >= 'a' && *src <= 'f' ? *src - 'a' + 10 : *src - 'A' + 10));
                        src++;
                    }
                    /* code to handle hexadecimal number */
                }
                else
                {
                    src--;
                    while (*src >= '0' && *src <= '9')
                    {
                        token_val = (token_val * 10) + (*src - '0');
                        src++;
                    }
                    /* code to handle decimal number */
                }
                tkn = TOKEN_NUM; // Set the token type
                return; // Exit after processing the number
            }
            else if ((tkn >= 'a' && tkn <= 'z') || (tkn >= 'A' && tkn <= 'Z') || tkn == '_')
            {
                char* last_pos = src - 1;
                long hash = tkn;
                while ((*src >= '0' && *src <= '9') || (*src >= 'a' && *src <= 'z') || (*src >= 'A' && *src <= 'Z') || *src == '_')
                {
                    hash = (hash * 0x40) + *src;
                    src++;
                }
                if (glob_stack_find(last_pos))
                {
                    /* code to handle existing identifier */
                    tkn = glob_stack_get(last_pos).tkn; // Set the token type from the existing identifier
                    return; // Exit after processing the existing identifier
                }
                /* code to handle identifiers */
                glob_data_t item;
                item.hash = hash;
                item.data = NULL; // Initialize data pointer if needed
                
                tkn = item.tkn = TOKEN_ID; // Set the token type
                glob_stack_set(last_pos, item);
                return; // Exit after processing the identifier
            }
            else
            {
                /* code to handle other tokens */
            }
            break;
        }
    }
    
}