/*
 * tcp_nodelay_preload.c — LD_PRELOAD shim that forces TCP_NODELAY on every
 * accepted socket.
 *
 * Why: the bundled smg HTTP gateway (a prebuilt Rust `.so`) does not set
 * TCP_NODELAY on the client-facing sockets it accepts. With HTTP keep-alive
 * (every real OpenAI/aiohttp client reuses connections), the first streamed
 * SSE chunk of each response is a sub-MSS write that Nagle's algorithm holds
 * until the peer's delayed ACK (~40 ms on Linux) arrives — inflating
 * single-stream TTFT by ~40 ms at concurrency 1. Disabling Nagle on the
 * accepted socket removes the stall without affecting throughput.
 *
 * smg resolves accept4()/accept() dynamically from libc, so intercepting them
 * here lets us set the option on the fd the gateway will actually stream over.
 * Applied via LD_PRELOAD to the gateway subprocess only (see cli/_proc.py).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

static int (*real_accept4)(int, struct sockaddr *, socklen_t *, int) = 0;
static int (*real_accept)(int, struct sockaddr *, socklen_t *) = 0;

static void set_nodelay(int fd) {
    if (fd < 0) return;
    int one = 1;
    /* Best-effort: harmless (EOPNOTSUPP) on non-TCP sockets. */
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}

int accept4(int sockfd, struct sockaddr *addr, socklen_t *addrlen, int flags) {
    if (!real_accept4)
        real_accept4 = (int (*)(int, struct sockaddr *, socklen_t *, int))
            dlsym(RTLD_NEXT, "accept4");
    int fd = real_accept4(sockfd, addr, addrlen, flags);
    set_nodelay(fd);
    return fd;
}

int accept(int sockfd, struct sockaddr *addr, socklen_t *addrlen) {
    if (!real_accept)
        real_accept = (int (*)(int, struct sockaddr *, socklen_t *))
            dlsym(RTLD_NEXT, "accept");
    int fd = real_accept(sockfd, addr, addrlen);
    set_nodelay(fd);
    return fd;
}
