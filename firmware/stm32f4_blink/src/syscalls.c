#include <errno.h>
#include <sys/stat.h>

/* Heap statique pour newlib-nano snprintf (%.4f nécessite malloc interne) */
#define HEAP_SIZE 512U
static char _heap[HEAP_SIZE];
static char *_heap_ptr = _heap;

void *_sbrk(int incr)
{
    if (_heap_ptr + incr > _heap + HEAP_SIZE) { errno = ENOMEM; return (void *)-1; }
    char *prev = _heap_ptr;
    _heap_ptr += incr;
    return (void *)prev;
}

int _close(int fd)                    { (void)fd; return -1; }
int _fstat(int fd, struct stat *st)   { (void)fd; st->st_mode = S_IFCHR; return 0; }
int _isatty(int fd)                   { (void)fd; return 1; }
int _lseek(int fd, int off, int wh)   { (void)fd; (void)off; (void)wh; return 0; }
int _read(int fd, char *buf, int len) { (void)fd; (void)buf; (void)len; return 0; }
int _getpid(void)                     { return 1; }
int _kill(int pid, int sig)           { (void)pid; (void)sig; errno = EINVAL; return -1; }

/* Requis par -u _printf_float (nano.specs) pour activer snprintf("%.4f") */
int _write(int fd, const char *buf, int len) { (void)fd; (void)buf; return len; }
__attribute__((noreturn)) void _exit(int status) { (void)status; while (1) {} }
