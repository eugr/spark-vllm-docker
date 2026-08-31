#define _GNU_SOURCE

#include <errno.h>
#include <linux/aio_abi.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

struct ple_aio {
    aio_context_t context;
    unsigned entries;
    struct iocb *requests;
    struct iocb **request_ptrs;
    struct io_event *events;
};

static _Thread_local char ple_error[256];

static void set_error(const char *where, int value) {
    snprintf(ple_error, sizeof(ple_error), "%s: %s", where, strerror(value));
}

const char *ple_aio_error(void) { return ple_error; }

void ple_aio_destroy(struct ple_aio *aio);

struct ple_aio *ple_aio_create(unsigned entries) {
    struct ple_aio *aio;
    long result;

    ple_error[0] = '\0';
    if (entries == 0) {
        set_error("entries", EINVAL);
        return NULL;
    }
    aio = calloc(1, sizeof(*aio));
    if (aio == NULL) {
        set_error("calloc", errno);
        return NULL;
    }
    aio->entries = entries;
    aio->requests = calloc(entries, sizeof(*aio->requests));
    aio->request_ptrs = calloc(entries, sizeof(*aio->request_ptrs));
    aio->events = calloc(entries, sizeof(*aio->events));
    if (aio->requests == NULL || aio->request_ptrs == NULL
        || aio->events == NULL) {
        set_error("calloc arrays", errno);
        ple_aio_destroy(aio);
        return NULL;
    }
    result = syscall(__NR_io_setup, entries, &aio->context);
    if (result < 0) {
        set_error("io_setup", errno);
        ple_aio_destroy(aio);
        return NULL;
    }
    return aio;
}

void ple_aio_destroy(struct ple_aio *aio) {
    if (aio == NULL) {
        return;
    }
    if (aio->context != 0) {
        syscall(__NR_io_destroy, aio->context);
    }
    free(aio->events);
    free(aio->request_ptrs);
    free(aio->requests);
    free(aio);
}

int ple_aio_read_rows(struct ple_aio *aio, const int *fds,
                      const int64_t *offsets, void *output,
                      unsigned count, unsigned row_bytes) {
    unsigned submitted = 0;
    unsigned completed = 0;
    int first_error = 0;

    ple_error[0] = '\0';
    if (aio == NULL || fds == NULL || offsets == NULL || output == NULL
        || count == 0 || row_bytes == 0) {
        set_error("arguments", EINVAL);
        return -EINVAL;
    }
    if (count > aio->entries) {
        set_error("batch exceeds context", E2BIG);
        return -E2BIG;
    }
    memset(aio->requests, 0, count * sizeof(*aio->requests));
    for (unsigned request = 0; request < count; ++request) {
        struct iocb *iocb = &aio->requests[request];
        iocb->aio_data = (uint64_t)request + 1;
        iocb->aio_lio_opcode = IOCB_CMD_PREAD;
        iocb->aio_fildes = (uint32_t)fds[request];
        iocb->aio_buf = (uint64_t)(uintptr_t)((char *)output
                                              + (size_t)request * row_bytes);
        iocb->aio_nbytes = row_bytes;
        iocb->aio_offset = offsets[request];
        aio->request_ptrs[request] = iocb;
    }

    while (submitted < count) {
        long result = syscall(__NR_io_submit, aio->context,
                              count - submitted,
                              aio->request_ptrs + submitted);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error("io_submit", errno);
            return -errno;
        }
        if (result == 0) {
            set_error("io_submit submitted zero requests", EIO);
            return -EIO;
        }
        submitted += (unsigned)result;
    }

    while (completed < count) {
        long result = syscall(__NR_io_getevents, aio->context, 1,
                              count - completed, aio->events, NULL);
        if (result < 0) {
            if (errno == EINTR) {
                continue;
            }
            set_error("io_getevents", errno);
            return -errno;
        }
        for (long index = 0; index < result; ++index) {
            struct io_event *event = &aio->events[index];
            unsigned request = (unsigned)event->data - 1;
            if (request >= count && first_error == 0) {
                first_error = -EPROTO;
            } else if (event->res != (int64_t)row_bytes && first_error == 0) {
                first_error = event->res < 0 ? (int)event->res : -EIO;
            }
        }
        completed += (unsigned)result;
    }
    if (first_error != 0) {
        set_error("read completion", -first_error);
        return first_error;
    }
    return 0;
}
