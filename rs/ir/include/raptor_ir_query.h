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

/* Whether an accumulate under the contract commits through a companion plane (*out is 0 or 1): the compensated contract and
   the one a keyed accumulate takes by default. */
int rx_compensates(rx_ctx* ctx, const char* contract, int* out);

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
   (role, name) pairs, in declaration order, such as (('vec_in', 'x'), ('mat_in', 'm')). Answers RX_OK for a sound one. */
int rx_quantity_check(rx_ctx* ctx, const char* slug, const char* pack, const char* wires_kt);

#ifdef __cplusplus
}
#endif
#endif
