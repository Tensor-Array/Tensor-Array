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
#include "vm_instruction.h"
#include "vmop.h"
#include "vm.h"

VM_INSTRUCTION* orig = NULL;
VM_INSTRUCTION* pc = NULL;

void eval()
{
    printf("vmstart\n");
    VM_INSTRUCTION_V2 op;
    pc = orig;
    while (1)
    {
        op = *((VM_INSTRUCTION_V2*)pc++);
        printf("vmopassign %ld %ld %ld \n", orig, pc, op);
        switch (op)
        {
            case LEA:
                // Load effective address
                break;
            case IMM:
                // Immediate value
                any_type = *pc++;
                any_value = *pc++;
                op_imm();
                break;
            case JMP:
                // Jump to address
                pc = (VM_INSTRUCTION*) *pc;
                break;
            case CALL:
                // Function call
                break;
            case JZ:
                // Jump if zero
                pc = (any_value) ? (VM_INSTRUCTION*)*pc : pc + 1;
                break;
            case JNZ:
                // Jump if not zero
                pc = (any_value) ? pc + 1 : (VM_INSTRUCTION*)*pc;
                break;
            case ENT:
                // Enter function
                break;
            case ADJ:
                // Adjust stack pointer
                break;
            case LEV:
                // Leave function
                break;
            case RET:
                // Return from function
                return;
            case LI:
                // Load integer
                break;
            case LC:
                // Load character
                break;
            case SI:
                // Store integer
                break;
            case SC:
                // Store character
                break;
            case SET:
                // Set value
                op_set();
                break;
            case GET:
                // Get value
                op_get();
                break;
            case PUSH:
                // Push value onto stack
                op_push();
                break;
            case PTR_PUSH:
                // Push value onto stack
                op_ptr_push();
                break;
            case GETELEM:
                // Get element from array
                break;
            case SETELEM:
                // Set element in array
                break;
            case ADDELEM:
                // Add element to array
                break;
            case OR:
                // Logical OR operation
                op_or();
                break;
            case XOR:
                // Logical XOR operation
                break;
            case AND:
                // Logical AND operation
                op_and();
                break;
            case EQ:
                // Equality check
                op_eq();
                break;
            case NE:
                // Not equal check
                op_ne();
                break;
            case LT:
                // Less than check
                op_lt();
                break;
            case GT:
                // Greater than check
                op_gt();
                break;
            case LE:
                // Less than or equal check
                op_le();
                break;
            case GE:
                // Greater than or equal check
                op_ge();
                break;
            case ADD:
                // Addition operation
                op_add();
                break;
            case SUB:
                // Subtraction operation
                op_sub();
                break;
            case MUL:
                // Multiplication operation
                op_mul();
                break;
            case DIV:
                // Division operation
                op_div();
                break;
            case MATMUL:
                // Matrix multiplication operation
                op_matmul();
                break;
            case POS:
                // Unary plus operation
                op_pos();
                break;
            case NEG:
                // Unary minus operation
                op_neg();
                break;
            case NOT:
                // Logical NOT operation
                op_not();
                break;
            case SHL:
                // Shift left operation
                break;
            case SHR:
                // Shift right operation
                break;
            case OPEN:
                // Open file operation
                op_open();
                break;
            case READ:
                // Read from file operation
                break;
            case CLOSE:
                // Close file operation
                break;
            case PRTF:
                // Print formatted output
                break;
            case MALC:
                // Memory allocation operation
                break;
            case MSET:
                // Memory set operation
                break;
            case MCMP:
                // Memory compare operation
                break;
            case EXIT:
                // Exit operation
                op_exit();
                return;
            default:
                fprintf(stderr, "Unknown instruction: %d\n", op);
                exit(1);
        }
    }
}
