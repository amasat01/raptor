/* The analyses of the IR a front end asks about nodes it made (a derivative pass reads them). The binding is generated from raptor_ir.h,
   raptor_ir_front.h and this header. Every call returns a status (0 ok, 1 refused, 2 bad argument, 3 buffer too small, 4 internal
   failure); a refusal's text is rx_last_error(ctx). Lists use the caller-buffer protocol of raptor_ir.h (*n is the size needed). */
#ifndef RAPTOR_WINGS_FRONT_H
#define RAPTOR_WINGS_FRONT_H
#include <stddef.h>
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The base of an add_at chain in *base and its links as (index, value) pairs flattened into buf, in the order they were added; *n counts the nodes written (twice the links). A node that is no add_at is its own base. */
int rw_slotwise_chain(rx_ctx* ctx, rx_node node, rx_node* base, rx_node* buf, size_t cap, size_t* n);
/* Whether the node is a vec or splat of zero literals (*out is 0 or 1). */
int rw_slotwise_is_structural_zero(rx_ctx* ctx, rx_node node, int* out);

#ifdef __cplusplus
}
#endif
#endif
