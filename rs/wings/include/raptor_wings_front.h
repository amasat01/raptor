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

/* The roots in slotwise form, one for one in buf: a wide value built from vec/splat becomes a vec of scalar parts with the structural zeros
   folded (component, sum, add/sub/neg/mul/div, select and dispatch per slot; an add_at chain over a slot-form base stays a chain and is
   lowered to selects where a static slot is forced). A root may be a value, a sink (rebuilt over its rewritten value) or a loop (never entered);
   a node with nothing to rewrite comes back as the same node. The wing kinds must be declared (rw_declare_kinds) before a dispatch is pinned to a strategy. */
int rw_slotwise(rx_ctx* ctx, const rx_node* roots, size_t nroots, rx_node* buf, size_t cap, size_t* n);

/* The sinks with each ordered accumulate replaced by the two log stores of its slot (the list in buf can be longer than the sinks); a site in
   a loop takes one slot per iteration. Refuses, with the reason, an accumulate the log cannot hold (nested loops, a step other than 1, a
   value that is not a scalar). The wing kinds must be declared (rw_declare_kinds). */
int rw_logs_ordered(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, rx_node* buf, size_t cap, size_t* n);

/* The strategies a dispatch can be pinned to: "switch", then one for each wing kind that refines dispatch (<strategy>_dispatch) in the order of the
   wing manifests, as the names protocol. */
int rw_dispatch_strategies(rx_ctx* ctx, char* buf, size_t cap, size_t* len, size_t* count);

/* The strategy word of each dispatch group of the walk in the walk's order ("switch", "predicated", "segmented" or "auto"), as the names protocol;
   the nodes are rx_dispatch_nodes', the group's name is dispatch<index>. */
int rw_walk_strategies(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len, size_t* count);

/* The canonical walk of a kernel's sinks, a walk number like rx_canonical's: over declared_kt (the key text of a tuple of Slot records), the
   compensation slot <name>_comp of every summing sample_sum (refused, by name, when a sink already holds a plane of that role and name),
   the finished-count slot when a sink finishes, and the branch-mix slot <group>_mix of every dispatch group that counts its branch mix; the
   fused-steps plane is exempt from the access class. force_access is NULL for none. */
int rw_cells_walk(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, const char* declared_kt, const char* force_access /* NULL */, rx_walk* out);

/* The cell sinks that store a reduction over samples to the plane name, in buf: one for a scalar, one per component (planes name_0, name_1, ...)
   for a vector or matrix, or the keyed accumulates of a reduction over some lane axes. */
int rw_cells_stored(rx_ctx* ctx, rx_node reduction, const char* name, rx_node* buf, size_t cap, size_t* n);

/* The sinks with every block-sum write replaced by the per-entry accumulates of its keyed form (an all-lane block is at key 0 of 1), in buf; any
   other sink stays. The wing kinds must be declared (rw_declare_kinds). */
int rw_blocks_expand_stores(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, rx_node* buf, size_t cap, size_t* n);

/* The factors of a keyed-sum payload that is a sum over per-row rank-1 products: u and v, in buf (an outer product, a vector times a
   scalar, or the product of a transposed matrix with itself or with a vector of its rows); no node for any other payload. */
int rw_block_factors(rx_ctx* ctx, rx_node payload, rx_node* buf, size_t cap, size_t* n);
/* How many rows a lane holds of a payload that has factors (1 for an outer product or a vector times a scalar); refuses a payload that has none. */
int rw_block_factor_rows(rx_ctx* ctx, rx_node payload, int64_t* out);
/* Whether the ordered contract keeps the payload (a floating scalar or vector whose extents are all at most the limit) as one block (*out is 0 or 1).
   limit holds no entry (any extent) or one (the largest extent). */
int rw_block_ordered_case(rx_ctx* ctx, rx_node payload, const char* contract, const int64_t* limit, size_t nlimit, int* out);
/* Whether the keyed sum of the payload can be one block by its structure alone: ordered_case, or a float payload within the limit that has factors. */
int rw_block_shaped(rx_ctx* ctx, rx_node payload, const char* contract, const int64_t* limit, size_t nlimit, int* out);

/* Declares in ctx every kind the wings declare (the base and simt manifests, embedded in this library), which the rewrites of this
   library make nodes of (a box store, for one). Call it once on a context before the first rewrite or before making a node of such a kind; it
   may be called again, and after a unit has declared the same kinds, with no effect. */
int rw_declare_kinds(rx_ctx* ctx);

#ifdef __cplusplus
}
#endif
#endif
