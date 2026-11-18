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
/* Whether one trip of the loop is safe to run and discard for a stopped lane (no computed-index read, switch, raw block or integer division) (*out is 0 or 1). */
int rw_loops_speculative_step(rx_ctx* ctx, rx_node loop, int* out);

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




/* The sinks of a kernel in the warp form (one warp serves a sample; every wide value is held sliced across the warp's lanes: element j of
   lane l is element j * warp + l), made in ctx, whose dialect must declare warp_lane and warp_store. A value is sliced when it is a vector
   of a whole number of warps and at least slice_min long. *found is 0 when a node has no rule in the form, or the automatic choice (explicit
   0) declines the kernel (only run-time indexed writes, or nothing sliced wider than team_min); then nothing is returned. A commit becomes a warp
   store, a sum, dot or norm of sliced values a sum over the lanes of each lane's partial sum. buf must hold nsinks nodes (else RX_TRUNC,
   nothing made). The caller takes the walk again over the result. */
int rw_lanes_slice_sinks(rx_ctx* ctx, const rx_node* sinks, size_t nsinks, int64_t warp, int64_t slice_min, int64_t team_min, int explicit,
                         rx_node* buf, size_t cap, size_t* count, int* found);

#ifdef __cplusplus
}
#endif
#endif
