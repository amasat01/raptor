/* The C door of the IR core, the QUERY half: what a backend asks of a built context (the BUILD half is raptor_ir.h, which
   defines the types and status codes used here; the typed reads of a node, which a front end shares, are raptor_ir_front.h,
   included here). Every call returns a status; the text of a refusal is rx_last_error(ctx). */
#ifndef RAPTOR_IR_QUERY_H
#define RAPTOR_IR_QUERY_H
#include "raptor_ir.h"
#include "raptor_ir_front.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Whether a node lies inside a loop body of the walk (*out is 0 or 1): the flat walk gives such a
    node no position. Refuses a
   node or walk the context does not hold. */
int rx_in_loop_body(rx_ctx* ctx, rx_walk walk, rx_node node, int* out);

int rx_walk_order(rx_ctx* ctx, rx_walk walk, rx_node* buf, size_t cap, size_t* n);

/* Whether the walk binds the slot (role, name): *out is 1 when its argument list holds that pair, else 0. */
int rx_walk_binds(rx_ctx* ctx, rx_walk walk, const char* role, const char* name, int* out);

/* The walk's argument list as (role, name) pairs in argument order: *count pairs, role then name, each
   NUL-terminated, back to back in buf; *len is the bytes they take with their NULs (the text protocol:
   cap is at least *len + 1). */
int rx_walk_args(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len, size_t* count);

/* The canonical position of any node the walk's traversal reached (a duplicate answers its representative's position). */
int rx_walk_position(rx_ctx* ctx, rx_walk walk, rx_node node, uint32_t* pos);

/* The nodes of a loop's body, post-order, by record identity (boundary values and the loop's placeholders excluded). */
int rx_body_nodes(rx_ctx* ctx, rx_node loop, rx_node* buf, size_t cap, size_t* n);

/* What a loop's body evaluates, in emission order: carried results, sink operands, then the break condition and values. */
int rx_body_roots(rx_ctx* ctx, rx_node loop, rx_node* buf, size_t cap, size_t* n);

/* The node a reference to node stands for (a primitive stands for its forward). */
int rx_through(rx_ctx* ctx, rx_node node, rx_node* out);

/* Whether two index expressions are the same value (out is 0 or 1). */
int rx_same_index(rx_ctx* ctx, rx_node a, rx_node b, int* out);

/* Whether a node is of the kind named (a kind name as the manifest lists it) or of a kind that
   refines it. */
int rx_is_a(rx_ctx* ctx, rx_node node, const char* name, int* out);

/* The dispatch nodes of a walk, top-level first, then those inside loop bodies: the nodes its
   `dispatches` listing names, in its order. */
int rx_dispatch_nodes(rx_ctx* ctx, rx_walk walk, rx_node* buf, size_t cap, size_t* n);

/* A constant raptor owns that is a list of strings (a string constant answers its one text): *count texts, each NUL-terminated, back to back in buf; *len is
   the bytes they take with their NULs (the text protocol: cap is at least *len + 1). A name that is no such
   list is a refusal. */
int rx_const_names(rx_ctx* ctx, const char* name, char* buf, size_t cap, size_t* len, size_t* count);

#ifdef __cplusplus
}
#endif
#endif
