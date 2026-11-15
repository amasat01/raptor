/* The C door of the IR core, the FRONT-END half: what a front end may READ of the nodes it made: a node's class, operands, type
   and typed fields. The binding is generated from the build half (raptor_ir.h) and this one; a backend reads these through
   raptor_ir_query.h, which includes this header. Every call returns a status; the text of a refusal is rx_last_error(ctx). */
#ifndef RAPTOR_IR_FRONT_H
#define RAPTOR_IR_FRONT_H
#include "raptor_ir.h"
#ifdef __cplusplus
extern "C" {
#endif

/* The class name of a node (the text protocol of raptor_ir.h). */
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

/* The operands of a node, in order (two-call array protocol: *n is the count, RX_TRUNC when cap is smaller). */
int rx_operands(rx_ctx* ctx, rx_node node, rx_node* buf, size_t cap, size_t* n);

#ifdef __cplusplus
}
#endif
#endif
