typedef enum
{
    TOKEN_NUM = 0x80, TOKEN_SYS, TOKEN_GLO, TOKEN_LOC, TOKEN_ID,
    TOKEN_FUNC, TOKEN_ELSE, TOKEN_ENUM, TOKEN_IF, TOKEN_RETURN, TOKEN_SIZEOF,
    TOKEN_WHILE, TOKEN_ASSIGN, TOKEN_COND, TOKEN_LOR, TOKEN_LAN,
    TOKEN_OR, TOKEN_XOR, TOKEN_AND, TOKEN_SHL, TOKEN_SHR,
    TOKEN_EQ, TOKEN_NE, TOKEN_LT, TOKEN_GT, TOKEN_LE, TOKEN_GE,
    TOKEN_ADD, TOKEN_SUB, TOKEN_MUL, TOKEN_DIV, TOKEN_MATMUL, TOKEN_POS, TOKEN_NEG, TOKEN_NOT
} TOKEN_TYPE;

char* tknname[] = {
    "num", "sys", "glo", "loc", "id",
    "func", "else", "enum", "if", "return", "sizeof",
    "while", "assign", "cond", "lor", "lan",
    "or", "xor", "and",
    "eq", "ne", "lt", "gt", "le", "ge",
    "add", "sub", "mul", "div", "matmul", "pos", "neg", "not", "brak"
};

void token_next();
extern long tkn = 0;
