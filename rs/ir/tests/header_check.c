#include "../include/raptor_ir.h"
int main(void) { rx_ctx *c = rx_ctx_new(); rx_ctx_free(c); return (int)rx_abi_version() + RX_PANIC; }
