#include <stdio.h>
#include <stdlib.h>
#include "vmop.h"
#include "vm.h"

VM_INSTRUCTION* orig;
void** pc;
void* any_value;

void eval()
{
    VM_INSTRUCTION op;
    pc = orig;
    while (1)
    {
        
        /* code */
        op = *pc++;
        switch (op)
        {
            case LEA:
                // Load effective address
                break;
            case IMM:
                // Immediate value
                any_value = *pc++;
                break;
            case JMP:
                // Jump to address
                pc = *pc;
                break;
            case CALL:
                // Function call
                break;
            case JZ:
                // Jump if zero
                pc = (any_value) ? pc + 1 : *pc;
                break;
            case JNZ:
                pc = (any_value) ? *pc : pc + 1;
                // Jump if not zero
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
