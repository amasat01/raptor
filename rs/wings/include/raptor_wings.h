/* The analyses the writers ask about a lowered walk. Every call returns a status (0 ok, 1 refused, 2 bad argument, 3 buffer too
   small, 4 internal failure); a refusal's text is rx_last_error(ctx) where a context is given. Text and list outputs use the
   caller-buffer protocol of raptor_ir.h (buf, cap, *len or *n: the size needed; RX_TRUNC when cap is too small). */
#ifndef RAPTOR_WINGS_H
#define RAPTOR_WINGS_H
#include <stddef.h>
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

/* Whether the lanes of a tile may run side by side over the walk (*out is 0 or 1). */
int rw_cpu_lane_form_access(rx_ctx* ctx, rx_walk walk, int* out);
/* The top-level loops the host runs interchanged, in walk order; a loop folding a vector longer than unroll_limit (the most trips the host compiler unrolls) is not one. */
int rw_cpu_interchanged_loops(rx_ctx* ctx, rx_walk walk, int64_t unroll_limit, rx_node* buf, size_t cap, size_t* n);
/* Whether the node is a call of a library function (*out is 0 or 1). */
int rw_cost_is_library_call(rx_ctx* ctx, rx_node node, int* out);
/* Whether the node is a linear-algebra factorisation, consumer or projection, which branches per lane (*out is 0 or 1). */
int rw_linalg_is_linalg(rx_ctx* ctx, rx_node node, int* out);
/* Whether a dispatch node renders as a switch (*out is 0 or 1). */
int rw_dispatch_switch_form(rx_ctx* ctx, rx_node node, int* out);
/* Whether the walk promotes the lookup plane of that name to a parameter cell (*out is 0 or 1). */
int rw_ir_param_plane(rx_ctx* ctx, rx_walk walk, const char* name, int* out);
/* The accumulates of the walk that keep a companion correction plane, in walk order, once per name (the caller names the plane). */
int rw_ir_companions(rx_ctx* ctx, rx_walk walk, rx_node* buf, size_t cap, size_t* n);
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
int rw_fold_rolled(rx_ctx* ctx, rx_node node, const int64_t* fold_min, const char* form, int* out);
/* The key run of one keyed sink of the kernel with the given sinks: found is 0 for none; run_max and block_run are the target's lane numbers. */
enum { RW_KEY_PROVEN = 1, RW_KEY_SORTED = 2, RW_KEY_TESTED = 3 };
typedef struct { int found; int form; int64_t width; int block; } rw_key_run;
int rw_lanes_key_run_in_kernel(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, rx_node sink, int64_t run_max, int64_t block_run, rw_key_run* out);
/* The target of a compensated sink: found is 0 when sink_id is not "compensated" (the name is then
    empty), else the target's name is
   written to name_buf (text protocol: *len, cap) and own_column is 0 or 1. returned_slot is NULL
   unless the kernel returns a typed
   value; output_names holds the declared output names, one per line (empty for none). The name is
   written before *out is set, so
   a RX_TRUNC answer leaves *out untouched: call again with a larger buffer. */
typedef struct { int found; int own_column; } rw_compensated_target;
int rw_split_compensated_target(rx_ctx* ctx, const char* sink_id, const char* compensation, const char* returned_slot,
                                const char* output_names, const rx_node* sinks, size_t nsinks,
                                rw_compensated_target* out, char* name_buf, size_t cap,
                                size_t* len);
/* Whether one trip of the loop is safe to run and discard for a stopped lane (no computed-index read, switch, raw block or integer division) (*out is 0 or 1). */
int rw_loops_speculative_step(rx_ctx* ctx, rx_node loop, int* out);

/* Whether the node names a plane a placement writes (a mutable's own or a scratch plane) (*out is 0 or 1). */
int rw_wide_is_scratch(rx_ctx* ctx, rx_node node, int* out);
/* Whether the node is the launch-start value of a vector mutable (*out is 0 or 1). */
int rw_wide_is_prior(rx_ctx* ctx, rx_node node, int* out);
/* How element k of a rank-1 node is read. */
enum { RW_ELEMENT_NONE = 0, RW_ELEMENT_PROXY = 1, RW_ELEMENT_INPUT = 2, RW_ELEMENT_PARTS = 3, RW_ELEMENT_SPLAT = 4, RW_ELEMENT_ELEMENTWISE = 5 };
int rw_wide_element_kind(rx_ctx* ctx, rx_node node, int* kind);
/* Whether a vector is reads of one plane at base + start + k * step: *found is 0 or 1; *base is UINT32_MAX when the index has no runtime part. */
int rw_wide_gather(rx_ctx* ctx, rx_node node, int* found, rx_node* plane, rx_node* base, int64_t* start, int64_t* step, int* clamped);
/* The nodes and everything inside their loop bodies, in order. */
int rw_loops_flat(rx_ctx* ctx, const rx_node* nodes, size_t n, rx_node* buf, size_t cap, size_t* count);
/* The sinks of a boxed entry: each sink once per lane of the box (lanes along lane axis `axis`, `stride` flat lanes apart), as
   copies made in ctx, whose dialect must declare box_store. buf must hold nsinks * lanes nodes: otherwise RX_TRUNC with *count set
   and nothing made. A loop result that moves along the axis is refused (it needs the core's loop constructor). */
int rw_box_boxed_sinks(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, size_t axis, size_t lanes, int64_t stride,
                       rx_node* buf, size_t cap, size_t* count);
/* The sinks with every least-squares node (lstsq, lstsq_gram_solve, lstsq_info) lowered to a Cholesky factor of J^T J and its solve,
   or to QR when J has more columns than *max_dim; max_dim is NULL when the target states none (refused when a node needs it).
   *changed is 0 when the sinks come back as given. buf must hold nsinks nodes (else RX_TRUNC, nothing made). A least-squares node
   inside a loop body is refused (it needs the core's loop constructor). The caller takes the walk again over the result. */
int rw_lstsq_lowered(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, const int64_t* max_dim, rx_node* buf, size_t cap,
                     size_t* count, int* changed);

/* The sinks with each wide value that needs a plane placed in one: a `resident` op over the value and a leaf naming the plane, made in ctx;
   a vector of scaled reads first becomes a scaled run. A kernel the pass leaves alone (thread_form != 0, no sinks, nothing needing a plane)
   comes back as given and *changed is 0. The target's facts: runtime_min five numbers (read, slot_read, rotate, write, accum): the widest vector
   kept in registers when accessed at a run-time index in that pattern, wide_min the widest with only compile-time indices that stays there (the table of the build's scalar width),
   rolled_min and form (as rw_fold_rolled; rolled_min NULL is refused where a threshold is read), taken the plane names to avoid, one per
   line. buf must hold nsinks nodes (else RX_TRUNC, nothing made). Loops with placed slots are made again by the core's loop constructor. The caller takes the walk again over the result. */
int rw_wide_place(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, int thread_form, const int64_t* runtime_min,
                  int64_t wide_min, const int64_t* rolled_min, const char* form, const char* taken, rx_node* buf, size_t cap,
                  size_t* count, int* changed);


/* For each of the nodes whether it is proven equal over every aligned run of *warp consecutive lanes in the kernel with the given
   sinks (flags[i] is 0 or 1; warp is NULL when the target names no run width, so only what needs no run is proven). Proven: a
   constant, a uniform or nsamples value, a loop index, a read at a proven index, an op (not a random one) of proven operands, and
   floor(y / d) or y * (1 / d) of a lane form y when d is a multiple of the warp and a run differs only in y's fastest axis, with
   coefficient 1. Unknown is 0. flags must hold nnodes ints. */
int rw_lanes_warp_uniform(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, const int64_t* warp, const rx_node* nodes, size_t nnodes,
                          int* flags);

/* The sinks of a kernel in the warp form (one warp serves a sample; every wide value is held sliced across the warp's lanes: element j of
   lane l is element j * warp + l), made in ctx, whose dialect must declare warp_lane and warp_store. A value is sliced when it is a vector
   of a whole number of warps and at least slice_min long. *found is 0 when a node has no rule in the form, or the automatic choice (explicit
   0) declines the kernel (only run-time indexed writes, or nothing sliced wider than team_min); then nothing is returned. A commit becomes a warp
   store, a sum, dot or norm of sliced values a sum over the lanes of each lane's partial sum. buf must hold nsinks nodes (else RX_TRUNC,
   nothing made). The caller takes the walk again over the result. */
int rw_lanes_slice_sinks(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, int64_t warp, int64_t slice_min, int64_t team_min, int explicit,
                         rx_node* buf, size_t cap, size_t* count, int* found);

/* Declares in ctx every kind the wings declare (the base and simt manifests, embedded in this library), which the rewrites of this
   library make nodes of (a box store, for one). Call it once on a context before the first rewrite; it may be called again, and after
   a unit has declared the same kinds, with no effect. */
int rw_declare_kinds(rx_ctx* ctx);

#ifdef __cplusplus
}
#endif
#endif
