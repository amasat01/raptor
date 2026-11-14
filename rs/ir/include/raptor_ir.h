/* The C door of the IR core. Every call returns a status; the text of a refusal is rx_last_error(ctx).
   Text outputs use the caller-buffer protocol (buf, cap, *len): *len is the length needed, without the terminating NUL; the text is
   copied with a NUL when cap >= *len + 1, otherwise nothing is copied and the status is RX_TRUNC. A context is not thread safe. */
#ifndef RAPTOR_IR_H
#define RAPTOR_IR_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct rx_ctx rx_ctx;
typedef uint32_t rx_node;
typedef uint32_t rx_walk;
enum { RX_OK = 0, RX_REFUSED = 1, RX_BADARG = 2, RX_TRUNC = 3, RX_PANIC = 4 };

uint32_t rx_abi_version(void);
rx_ctx* rx_ctx_new(void);
void rx_ctx_free(rx_ctx* ctx);
const char* rx_last_error(const rx_ctx* ctx);

/* Declares the node classes of a dialect manifest (JSON) in this context. */
int rx_declare_dialect(rx_ctx* ctx, const char* manifest_json, size_t n, const char* first_party_kt);

/* Appends one node record and returns its handle; two equal calls give two handles. fields_kt is key text: a tuple of (name, value)
   pairs, a value being None, True, False, an int, a float, a string, a tuple, a list, a record Name(f=v, ...) or a node reference #h. */
int rx_make(rx_ctx* ctx, const char* cls, const char* dtype, const int64_t* shape, size_t rank, const char* tag /* NULL = None */,
            const rx_node* ops, size_t nops, const char* fields_kt, size_t n, rx_node* out);

int rx_class(rx_ctx* ctx, rx_node node, char* buf, size_t cap, size_t* len);
int rx_ttype(rx_ctx* ctx, rx_node node, char* buf, size_t cap, size_t* len);
int rx_operands(rx_ctx* ctx, rx_node node, rx_node* buf, size_t cap, size_t* n);
int rx_field(rx_ctx* ctx, rx_node node, const char* name, char* buf, size_t cap, size_t* len);

int rx_canonical(rx_ctx* ctx, const rx_node* sinks, size_t n, const char* declared_kt, const char* force_access /* NULL */,
                 const char* exempt_kt, rx_walk* out);
int rx_walk_order(rx_ctx* ctx, rx_walk walk, rx_node* buf, size_t cap, size_t* n);
int rx_walk_field(rx_ctx* ctx, rx_walk walk, const char* field, char* buf, size_t cap, size_t* len);
int rx_digest(rx_ctx* ctx, rx_walk walk, char out65[65]);

int rx_dump(rx_ctx* ctx, const rx_node* roots, size_t n, char* buf, size_t cap, size_t* len);
int rx_load(rx_ctx* ctx, const char* text, size_t n, rx_node* roots, size_t cap, size_t* nroots);
int rx_pseudo(rx_ctx* ctx, rx_node node, char* buf, size_t cap, size_t* len);

#ifdef __cplusplus
}
#endif
#endif
