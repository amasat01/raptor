/* The analyses of the IR a front end asks about nodes it made (a derivative pass reads them). The binding is generated from raptor_ir.h,
   raptor_ir_front.h and this header. Every call returns a status (0 ok, 1 refused, 2 bad argument, 3 buffer too small, 4 internal
   failure); a refusal's text is rx_last_error(ctx). Lists use the caller-buffer protocol of raptor_ir.h (*n is the size needed). */
#ifndef RAPTOR_WINGS_FRONT_H
#define RAPTOR_WINGS_FRONT_H
#include <stddef.h>
#include "raptor_ir.h"
#include "raptor_ir_front.h"
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

/* The names a wing owns, by name, as the names protocol: SPECIAL_KINDS (the op kinds that call the math library or draw a random number),
   FUSED_STEPS_PLANE and FINISHED_PLANE (the reserved lookup planes). Refuses any other name. */
int rw_wing_names(rx_ctx* ctx, const char* name, char* buf, size_t cap, size_t* len, size_t* count);

/* The strategies a dispatch can be pinned to: "switch", then one for each wing kind that refines dispatch (<strategy>_dispatch) in the order of the
   wing manifests, as the names protocol. */
int rw_dispatch_strategies(rx_ctx* ctx, char* buf, size_t cap, size_t* len, size_t* count);

/* The strategy word of each dispatch group of the walk in the walk's order ("switch", "predicated", "segmented" or "auto"), as the names protocol;
   the nodes are rx_dispatch_nodes', the group's name is dispatch<index>. */
int rw_walk_strategies(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len, size_t* count);

/* The dispatch groups of the walk in the walk's order as pairs (group name, name of its branch-mix slot): the second part is absent (kind 0) for a
   group that has no mix slot, which only an automatic group has. The two-call protocol of rx_field_pairs (*n is the count, RX_TRUNC when cap is smaller). */
int rw_walk_mix_slots(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);

/* The summing sample sums of the walk in the walk's order as pairs (the sum's plane, its compensation plane <name>_comp), the two-call protocol of
   rx_field_pairs. */
int rw_walk_companions(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);

/* The names a dispatch group owns, from the group's name, as the names protocol: the offsets plane and the class plane of a segmented group, its
   stale mask, and the branch-mix slot of an automatic group, in that order. */
int rw_group_plane_names(rx_ctx* ctx, const char* group, char* buf, size_t cap, size_t* len, size_t* count);

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

/* The Sample node written as At reads of its plane, in *out, the node that takes its place: nearest reads at floor(x + 0.5) on each axis, linear
   reads at floor(x) and the next whole index and weighs the 2, 4 or 8 corners by the products of (1 - w) and w; the edge rule is the At rule of each read.
   Refuses a node that is not a Sample. */
int rw_sample_expand(rx_ctx* ctx, rx_node node, rx_node* out);

/* Whether the node places a value in a plane (*out is 0 or 1). */
int rw_wide_is_resident(rx_ctx* ctx, rx_node node, int* out);
/* Whether the node names a scratch plane the runtime made (flat or warp-tiled), not a mutable's own (*out is 0 or 1). */
int rw_wide_is_tiled(rx_ctx* ctx, rx_node node, int* out);
/* How a placed plane is laid out. */
enum { RW_LAYOUT_PLANE_MAJOR = 0, RW_LAYOUT_WARP_TILED = 1 };
/* One placed plane: the leaf that names it, its layout and width. */
typedef struct { rx_node plane; int layout; int64_t width; } rw_plane_row;
/* The placements of the given nodes (loop bodies included), one row per plane, sorted by plane name; *count is the rows needed (RX_TRUNC when cap is too small). */
int rw_wide_record(rx_ctx* ctx, const rx_node* order, size_t n, rw_plane_row* rows, size_t cap, size_t* count);
/* Whether the node is spelled as its runtime twin; fold_min is NULL when the facts name no threshold, form is "auto", "rolled" or "static" (*out is 0 or 1). */
int rw_fold_rolled(rx_ctx* ctx, rx_node node, const int64_t* fold_min /* NULL */, const char* form, int* out);
/* The key run of one keyed sink of the kernel with the given sinks: found is 0 for none; run_max and block_run are the target's lane numbers. */
enum { RW_KEY_PROVEN = 1, RW_KEY_SORTED = 2, RW_KEY_TESTED = 3 };
typedef struct { int found; int form; int64_t width; int block; } rw_key_run;
int rw_lanes_key_run_in_kernel(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, rx_node sink, int64_t run_max, int64_t block_run, rw_key_run* out);
/* Whether the node names a plane a placement writes (a mutable's own or a scratch plane) (*out is 0 or 1). */
int rw_wide_is_scratch(rx_ctx* ctx, rx_node node, int* out);
/* Whether the node is the launch-start value of a vector mutable (*out is 0 or 1). */
int rw_wide_is_prior(rx_ctx* ctx, rx_node node, int* out);
/* How element k of a rank-1 node is read. */
enum { RW_ELEMENT_NONE = 0, RW_ELEMENT_PROXY = 1, RW_ELEMENT_INPUT = 2, RW_ELEMENT_PARTS = 3, RW_ELEMENT_SPLAT = 4, RW_ELEMENT_ELEMENTWISE = 5 };
int rw_wide_element_kind(rx_ctx* ctx, rx_node node, int* kind);
/* Whether a vector is reads of one plane at lead.., base + start + k * step: the running index is the last one of each read and lead[0..nlead] holds the
   index nodes ahead of it, shared by every read of the run (none for a plane read at one index; a plane has at most eight axes). found is 0 or 1;
   base is UINT32_MAX when the running index has no runtime part. Reads at several indices are a run only with the same lead nodes and the same edge rule. */
typedef struct { int found; rx_node plane; rx_node base; int64_t start; int64_t step; int clamped; size_t nlead; rx_node lead[8]; } rw_gather;
int rw_wide_gather(rx_ctx* ctx, rx_node node, rw_gather* out);
/* The sinks with each wide value that needs a plane placed in one: a `resident` op over the value and a leaf naming the plane, made in ctx;
   a vector of scaled reads first becomes a scaled run. A kernel the pass leaves alone (thread_form != 0, no sinks, nothing needing a plane)
   comes back as given and *changed is 0. The target's facts: runtime_min (nruntime = 5) five numbers (read, slot_read, rotate, write, accum): the widest vector
   kept in registers when accessed at a run-time index in that pattern, wide_min the widest with only compile-time indices that stays there (the table of the build's scalar width),
   rolled_min and form (as rw_fold_rolled; rolled_min NULL is refused where a threshold is read), taken the plane names to avoid, one per
   line. buf must hold nsinks nodes (else RX_TRUNC, nothing made). Loops with placed slots are made again by the core's loop constructor. The caller takes the walk again over the result. */
int rw_wide_place(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, int thread_form, const int64_t* runtime_min,
                  size_t nruntime, int64_t wide_min, const int64_t* rolled_min /* NULL */, const char* form, const char* taken,
                  rx_node* buf, size_t cap, size_t* count, int* changed);
/* For each of the nodes whether it is proven equal over every aligned run of *warp consecutive lanes in the kernel with the given
   sinks (flags[i] is 0 or 1; warp is NULL when the target names no run width, so only what needs no run is proven). Proven: a
   constant, a uniform or nsamples value, a loop index, a read at a proven index, an op (not a random one) of proven operands, and
   floor(y / d) or y * (1 / d) of a lane form y when d is a multiple of the warp and a run differs only in y's fastest axis, with
   coefficient 1. Unknown is 0. flags must hold nnodes ints. */
int rw_lanes_warp_uniform(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, const int64_t* warp /* NULL */, const rx_node* nodes, size_t nnodes,
                          int* flags);
/* Whether n exceeds the fold threshold (*out is 0 or 1); refused when fold_min is NULL, the facts naming no threshold. The rule is the
   library's: a caller never restates it. */
int rw_fold_exceeds(rx_ctx* ctx, int64_t n, const int64_t* fold_min /* NULL */, int* out);
/* The matrix input planes the walk reads only through a gather, sorted, as the names protocol: their column count is the caller's, not
   the lane count. */
int rw_walk_gather_mats(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len, size_t* count);

/* The target of a compensated sink: found is 0 when sink_id is not "compensated" (the name is then
    empty), else the target's name is
   written to name_buf (text protocol: *len, cap) and own_column is 0 or 1. returned_slot is NULL
   unless the kernel returns a typed
   value; output_names holds the declared output names, one per line (empty for none). The name is
   written before *out is set, so
   a RX_TRUNC answer leaves *out untouched: call again with a larger buffer. */
typedef struct { int found; int own_column; } rw_compensated_target;
int rw_split_compensated_target(rx_ctx* ctx, const char* sink_id, const char* compensation, const char* returned_slot /* NULL */,
                                const char* output_names, const rx_node* sinks, size_t nsinks,
                                rw_compensated_target* out, char* name_buf, size_t cap,
                                size_t* len);

/* Declares in ctx every kind the wings declare (the base and simt manifests, embedded in this library), which the rewrites of this
   library make nodes of (a box store, for one). Call it once on a context before the first rewrite or before making a node of such a kind; it
   may be called again, and after a unit has declared the same kinds, with no effect. */
int rw_declare_kinds(rx_ctx* ctx);

#ifdef __cplusplus
}
#endif
#endif
