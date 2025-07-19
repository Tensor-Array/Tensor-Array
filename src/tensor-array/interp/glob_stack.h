#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct
    {
        long tkn;
        long hash;
        long cls;
        void* data; // Pointer to additional data if needed
    } glob_data_t;
    void glob_data_set(char*, glob_data_t);
    glob_data_t glob_stack_get(char*);
    int glob_stack_find(char*);
#ifdef __cplusplus
}
#endif
