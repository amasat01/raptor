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

/* Whether the runtime zeroes the written plane (role, name) before the run (*out is 0 or 1): some sink on it accumulates, sums
   samples or is an ordered accumulate. A slot that is not a written plane, or is not bound, answers 0. */
int rx_walk_zeroed(rx_ctx* ctx, rx_walk walk, const char* role, const char* name, int* out);

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

/* Whether the runtime zeroes the written plane a sink node commits to before the run (out is 0 or 1): the one rule the walk's
   per-slot zeroed fact is built from. A node that is not a sink on a written plane answers 0. */
int rx_zeroes_plane(rx_ctx* ctx, rx_node node, int* out);

/* The dispatch nodes of a walk, top-level first, then those inside loop bodies: the nodes its
   `dispatches` listing names, in its order. */
int rx_dispatch_nodes(rx_ctx* ctx, rx_walk walk, rx_node* buf, size_t cap, size_t* n);

/* A constant raptor owns that is a list of strings (a string constant answers its one text): *count texts, each NUL-terminated, back to back in buf; *len is
   the bytes they take with their NULs (the text protocol: cap is at least *len + 1). A name that is no such
   list is a refusal. */
int rx_const_names(rx_ctx* ctx, const char* name, char* buf, size_t cap, size_t* len, size_t* count);

/* Whether the kind of a node states the feather `name` (*out is 0 or 1); a name that is no feather is a refusal. */
int rx_has_feather(rx_ctx* ctx, rx_node node, const char* name, int* out);

/* Whether a node reads a parameter cell: an At of a role constant labelled so (*out is 0 or 1). */
int rx_is_param_cell(rx_ctx* ctx, rx_node node, int* out);

/* The sink `sink` committing `value` instead (a commit, sample sum, keyed accumulate or finish, or a sink whose kind
   states the commits_value feather; any other sink is a refusal). */
int rx_with_value(rx_ctx* ctx, rx_node sink, rx_node value, rx_node* out);

/* The node where a matrix is wanted: a Cholesky factor becomes its projection, another factor is refused by name,
   any other node answers itself. */
int rx_as_matrix(rx_ctx* ctx, rx_node node, rx_node* out);

/* The terms of an add chain, left to right (the two-call array protocol); any other node is its own one term. A sub at
   the top alternates the sign of what follows, so only add is flattened. */
int rx_add_terms(rx_ctx* ctx, rx_node node, rx_node* buf, size_t cap, size_t* n);

/* Refuses, by name, a loop whose body holds a nested loop that leaves early, commits a sink or runs a recorded trip count
   (neither direction of differentiation can carry it); answers RX_OK otherwise. */
int rx_refuse_nesting(rx_ctx* ctx, rx_node loop);

/* Whether a loop is an accumulator (*out is 1: every carry's next value is an add chain holding that carry once as a bare
   term, read nowhere else) or a recurrence (0). Refuses a loop neither direction can carry, by name. */
int rx_loop_is_accumulator(rx_ctx* ctx, rx_node loop, int* out);

/* The addend a of each carry of an accumulator loop (c = c + a), in carry order (the two-call array protocol); the
   addends of one carry are summed left to right in the context. A recurrence is a refusal. */
int rx_loop_addends(rx_ctx* ctx, rx_node loop, rx_node* buf, size_t cap, size_t* n);

/* Whether the update commits through a companion plane (*out is 0 or 1): an add under the compensated summation. */
int rx_compensates(rx_ctx* ctx, rx_node node, int* out);

/* The read of the named parameter of the type (dtype, shape, tag; tag NULL = none): cell 0 of a one-cell lookup plane, or,
   with per_cell, the cell the sample's param_cell plane names. rx_is_param_cell answers 1 for the first form. */
int rx_param_cell(rx_ctx* ctx, const char* name, const char* dtype, const int64_t* shape, size_t rank,
                  const char* tag /* NULL */, int per_cell, rx_node* out);

/* The affine form of an integer index expression in the index variable var (NULL: the sample index): how the index moves with
   the variable. The answer is a list of texts (the names protocol): the one text "none" when no form is claimed, else "form",
   the coefficient "n/d", the offset "n/d" or "-" (not a compile-time integer), "1" when a bare literal scales it else "0", the
   symbol ("n_samples" when the launch's sample count scales it) or "-", then the declaration-named constants it is scaled
   by. A non-integer coefficient means the index depends on the variable only through floor(coefficient * s + offset).
   quotients: 0 refuses a division. Conservative: "none" claims nothing. */
int rx_index_form(rx_ctx* ctx, rx_node node, const rx_node* var /* NULL */, int quotients, char* buf, size_t cap,
                  size_t* len, size_t* count);

/* A number per node of an order (children before parents; the two-call array protocol into *buf): equal numbers mean the
   nodes are one value, so a value traced twice counts once. Two nodes are one value when their class, kind, type, literal
   (as the core's key text: nan equals nan, -0.0 is not 0.0, True is not 1), identity state, name and operand numbers are
   equal; an operand outside the order has no number. The numbers are dense from 0 in order of first appearance and mean
   something only within one call. */
int rx_structural_keys(rx_ctx* ctx, const rx_node* order, size_t n, int64_t* buf, size_t cap, size_t* count);

/* Refuses, by name, a compound quantity declaration: a pack outside the manifest's QUANTITY_PACKS, an empty slug, no wire, a
   wire whose role no leaf kind mints (QUANTITY_ROLE_PAIRS) or two wires of one name. wires_kt is the key text of a tuple of
   (role, name) pairs, in declaration order, such as (('plane_in', 'x'), ('plane_in', 'm')). Answers RX_OK for a sound one. */
int rx_quantity_check(rx_ctx* ctx, const char* slug, const char* pack, const char* wires_kt);

/* The derivative rules, as templates of the manifest run over the nodes of the context. A table is one of the manifest's rule
   tables (unary_d, bin_vjp, bin_jvp for rx_rule_factor; core, linalg, rays for the programs of rx_rule_vjp / rx_rule_jvp).
   Results come back as nodes in buf (the array protocol: *n is the count, RX_TRUNC when cap is smaller). */

/* The factor table/kind names over the nodes bound by name: bound is the names separated by white space, nodes[i] the node of
   the i-th name; the names in absent are bound to no value (an operand without a tangent). Answers one node for a factor, the
   live entries for bin_jvp (their sum is the tangent), the pairs (child, contribution) flattened for bin_vjp. */
int rx_rule_factor(rx_ctx* ctx, const char* table, const char* kind, const char* bound, size_t bound_len, const rx_node* nodes,
                   size_t nnodes, const char* absent, size_t absent_len, rx_node* buf, size_t cap, size_t* n);

/* The reverse contributions of node under the program table/kind, cot the incoming adjoint: pairs (child, adjoint) flattened. */
int rx_rule_vjp(rx_ctx* ctx, const char* table, const char* kind, rx_node node, rx_node cot, rx_node* buf, size_t cap, size_t* n);

/* The tangent of node under the program table/kind: tangent[i] is the tangent of the operand tangent_of[i], an operand not
   listed has none. Answers one node, or none when the rule yields no derivative. */
int rx_rule_jvp(rx_ctx* ctx, const char* table, const char* kind, rx_node node, const rx_node* tangent_of, const rx_node* tangent,
                size_t count, rx_node* buf, size_t cap, size_t* n);

/* The same two directions for a sum over samples read in the body (a ReduceSamples node); any other reduction is refused by
   name. The reverse answers the pair (operand, the sum over cot). */
int rx_reduce_vjp(rx_ctx* ctx, rx_node node, rx_node cot, rx_node* buf, size_t cap, size_t* n);
int rx_reduce_jvp(rx_ctx* ctx, rx_node node, const rx_node* tangent_of, const rx_node* tangent, size_t count, rx_node* buf,
                  size_t cap, size_t* n);

/* The kinds a rule table has templates for, in manifest order (the names protocol of rx_const_names). */
int rx_rule_kinds(rx_ctx* ctx, const char* table, char* buf, size_t cap, size_t* len, size_t* count);

/* The kinds of a rule mode, in table order: "zero" (no derivative) or "walker" (left to the transform). */
int rx_rule_modes(rx_ctx* ctx, const char* mode, char* buf, size_t cap, size_t* len, size_t* count);

/* kind over the nodes entry by entry when any is not rank-0, re-assembled with vec. */
int rx_entrywise(rx_ctx* ctx, const char* kind, const rx_node* nodes, size_t nnodes, rx_node* out);

/* A structural zero of the dtype and shape: a rank-0 constant, else a splat of it. */
int rx_zero_of(rx_ctx* ctx, const char* dtype, const int64_t* shape, size_t rank, rx_node* out);

/* Checks that the graph under roots is a CLOSED rule template of the primitive name: its free leaves are the names in allowed
   (separated by white space; inside a loop body too), a loop's index and carries are read only inside that loop, and nothing
   in it holds edges outside a loop, writes a plane (body sinks) or reads a tape. Answers RX_OK, or refuses with the first
   violation by name (rx_last_error). */
int rx_rule_template_check(rx_ctx* ctx, const char* name, const rx_node* roots, size_t nroots, const char* allowed,
                           size_t allowed_len);

/* Whether name is taken as a kind name, so a primitive may not be registered under it: the primitive kind, a core kind a
   declaration may refine, or a kind declared in the context (*out is 0 or 1). */
int rx_kind_reserved(rx_ctx* ctx, const char* name, int* out);

/* The type of the slot (role, name) the walk binds; refuses a slot it does not bind. */
int rx_walk_slot_type(rx_ctx* ctx, rx_walk walk, const char* role, const char* name, rx_type* out);

/* The names of the planes whose launch-start value the walk reads, sorted (the names protocol of rx_const_names). */
int rx_walk_prior(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len, size_t* count);

/* Whether the selector of the dispatch at index (the order of rx_dispatch_nodes) is static: no sink of the walk writes a plane it reads (*out is 0 or 1). */
int rx_walk_static_selector(rx_ctx* ctx, rx_walk walk, int64_t index, int* out);

/* The (role, name) slots the selector of the dispatch at index (the order of rx_dispatch_nodes) reads, sorted, as pairs (the two-call protocol of
   rx_field_pairs); refuses an index the walk lacks. */
int rx_walk_selector_reads(rx_ctx* ctx, rx_walk walk, int64_t index, rx_pair* buf, size_t cap, size_t* n);

/* The companion planes of the compensated keyed accumulates of the walk as pairs (companion plane <plane>_c, the plane it compensates), in the
   walk's order, once per companion (the two-call protocol of rx_field_pairs). */
int rx_walk_accum_companions(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);
/* The counting planes of the walk as pairs (count plane, the plane it counts), in the order the caller declared them: a Slot record's
   counts member states which plane a one-element count belongs to (the two-call protocol of rx_field_pairs). */
int rx_walk_counts(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);
/* The winner planes of the walk as pairs (winner plane, the plane it ranks), in the order the caller declared them: a Slot record's
   winner_of member states which plane a plane of lanes belongs to; each element of it names the lane whose value a minimum or a
   maximum kept, as an unsigned lane index, all ones for none (the two-call protocol of rx_field_pairs). */
int rx_walk_winners(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);
/* The wires of the quantities the walk reaches, in the walk's order, one pair (namespace, slot name of the wire) per wire in declaration
   order, the pairs of a quantity together; the two-call protocol of rx_field_pairs. */
int rx_walk_quantity_wires(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);
/* The slot numbers of those wires (the place of each in the walk's arguments): one pair (namespace, slot number) per wire, in exactly the
   order of rx_walk_quantity_wires. */
int rx_walk_quantity_slots(rx_ctx* ctx, rx_walk walk, rx_pair* buf, size_t cap, size_t* n);

/* The access class of the walk as text ("sample_local", "cross_sample_read", ...: the text protocol, cap is at least *len + 1). */
int rx_walk_access_class(rx_ctx* ctx, rx_walk walk, char* buf, size_t cap, size_t* len);

#ifdef __cplusplus
}
#endif
#endif
