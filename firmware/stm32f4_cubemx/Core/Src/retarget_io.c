#include "stm32f4xx_hal.h"
#include <errno.h>
#include <sys/stat.h>
#include <sys/unistd.h>

extern UART_HandleTypeDef huart3;   /* handle généré par CubeMX */

/* Retarget printf → USART3 (ST-LINK VCP) */
int _write(int file, char *ptr, int len)
{
    (void)file;
    HAL_UART_Transmit(&huart3, (uint8_t *)ptr, (uint16_t)len, HAL_MAX_DELAY);
    return len;
}

/* Stubs syscall minimaux requis par newlib-nano */
int _read(int file, char *ptr, int len)  { (void)file; (void)ptr; (void)len; return 0; }
int _close(int file)                     { (void)file; return -1; }
int _fstat(int file, struct stat *st)    { (void)file; st->st_mode = S_IFCHR; return 0; }
int _isatty(int file)                    { (void)file; return 1; }
int _lseek(int file, int ptr, int dir)   { (void)file; (void)ptr; (void)dir; return 0; }
int _getpid(void)                        { return 1; }
int _kill(int pid, int sig)              { (void)pid; (void)sig; errno = EINVAL; return -1; }

/* Heap pour printf (newlib malloc) — entre _end (BSS) et stack top */
extern char _end;
void *_sbrk(int incr)
{
    static char *heap = NULL;
    char *prev;
    if (heap == NULL) heap = &_end;
    prev  = heap;
    heap += incr;
    return (void *)prev;
}
