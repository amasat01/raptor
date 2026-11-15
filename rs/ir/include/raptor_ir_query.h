/* The C door of the IR core, the QUERY half: what a backend asks of a built context (the BUILD half is raptor_ir.h, which
   defines the types and status codes used here). Every call returns a status; the text of a refusal is rx_last_error(ctx). */
#ifndef RAPTOR_IR_QUERY_H
#define RAPTOR_IR_QUERY_H
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

int rx_class(rx_ctx* ctx, rx_node node, char* buf, size_t cap, size_t* len);

/* The type of a node: rank, extents, dtype code and tag (an empty text for none). Refuses a rank
    over RX_MAX_RANK or a tag of
   RX_TAG_CAP bytes or more. The dtype codes are the positions in the core's dtype list. */
enum { RX_MAX_RANK = 8, RX_TAG_CAP = 64 };
enum { RX_F64 = 0, RX_F32 = 1, RX_I32 = 2, RX_I64 = 3, RX_BOOL = 4, RX_INT32 = 5, RX_UINT32 = 6 };
typedef struct {
  uint32_t rank;
  int64_t dims[RX_MAX_RANK];
  int dtype;
  char tag[RX_TAG_CAP];
} rx_type;
int rx_ttype_of(rx_ctx* ctx, rx_node node, rx_type* out);

/* Typed field readers. A field that holds None answers RX_ABSENT (a valid question, no value); a
    field of another kind than the
   reader's, or a node without such a field, is a refusal. Text has no quotes. */
int rx_field_text(rx_ctx* ctx, rx_node node, const char* name, char* buf, size_t cap, size_t* len);
int rx_field_int(rx_ctx* ctx, rx_node node, const char* name, int64_t* out);
/* An integer field is read as its real value. */
int rx_field_real(rx_ctx* ctx, rx_node node, const char* name, double* out);
/* *out is 0 or 1. */
int rx_field_flag(rx_ctx* ctx, rx_node node, const char* name, int* out);
/* The nodes a field refers to, each distinct node once in order of first appearance, wherever they
    sit in the value (two-call
   array protocol: *n is the count, RX_TRUNC when cap is smaller). A field with no reference answers
   an empty array. */
int rx_field_nodes(rx_ctx* ctx, rx_node node, const char* name, rx_node* buf, size_t cap,
                   size_t* n);
/* The nodes a field refers to in the order they appear, a node listed as often as the field names
   it (a positional list; the same two-call array protocol). */
int rx_field_node_list(rx_ctx* ctx, rx_node node, const char* name, rx_node* buf, size_t cap,
                       size_t* n);
/* The kind of value a field holds (*out is one of the RX_KIND_ codes), so a caller picks the typed reader; a field that
   holds None answers RX_KIND_ABSENT with the status RX_OK. A list of integers is RX_KIND_INTS, of texts RX_KIND_NAMES, of
   references RX_KIND_NODES. */
enum {
  RX_KIND_ABSENT = 0,
  RX_KIND_TEXT = 1,
  RX_KIND_INTEGER = 2,
  RX_KIND_REAL = 3,
  RX_KIND_FLAG = 4,
  RX_KIND_NODES = 5,
  RX_KIND_NAMES = 6,
  RX_KIND_INTS = 7,
  RX_KIND_OTHER = 8
};
int rx_field_kind(rx_ctx* ctx, rx_node node, const char* name, int* out);
/* A tuple or list of integers (the two-call array protocol: *n is the count, RX_TRUNC when cap is smaller). */
int rx_field_ints(rx_ctx* ctx, rx_node node, const char* name, int64_t* buf, size_t cap, size_t* n);
/* A tuple or list of texts: *count names, each NUL-terminated, back to back in buf; *len is the
    bytes they take with their NULs
   (the text protocol: cap is at least *len + 1). */
int rx_field_names(rx_ctx* ctx, rx_node node, const char* name, char* buf, size_t cap,
                   size_t* len, size_t* count);

/* Whether a node lies inside a loop body of the walk (*out is 0 or 1): the flat walk gives such a
    node no position. Refuses a
   node or walk the context does not hold. */
int rx_in_loop_body(rx_ctx* ctx, rx_walk walk, rx_node node, int* out);

int rx_operands(rx_ctx* ctx, rx_node node, rx_node* buf, size_t cap, size_t* n);

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
