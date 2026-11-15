/* The C door of the IR core, the BUILD half: what a front end makes a context and its nodes with (what a front end reads back is raptor_ir_front.h, what a backend asks is raptor_ir_query.h). Every call returns a status; the text of a refusal is rx_last_error(ctx).
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
enum { RX_OK = 0, RX_REFUSED = 1, RX_BADARG = 2, RX_TRUNC = 3, RX_PANIC = 4,
       RX_ABSENT = 5 /* a valid question whose answer is that there is none */ };

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

/* Builds a node from constructor parameters (a tuple of (name, value) pairs; #i in a value is refs[i]): the declared program runs forward. */
int rx_new(rx_ctx* ctx, const char* cls, const char* params_kt, size_t n, const rx_node* refs, size_t nrefs, rx_node* out);

/* Builds an Op of a kind with its result type inferred from the operands; literal_kt is value text, empty for None. */
int rx_make_op(rx_ctx* ctx, const char* kind, const rx_node* ops, size_t nops, const char* literal_kt, size_t n, rx_node* out);

/* Builds an Op of a kind over the operands that carries the `literal` field of the node *like (NULL: no literal);
   the result type is inferred as rx_make_op does. */
int rx_make_op_like(rx_ctx* ctx, const char* kind, const rx_node* ops, size_t nops, const rx_node* like /* NULL */, rx_node* out);

int rx_canonical(rx_ctx* ctx, const rx_node* sinks, size_t n, const char* declared_kt, const char* force_access /* NULL */,
                 const char* exempt_kt, rx_walk* out);

/* The canonical walk over the sinks that keeps the slots the walk `like` binds (a sub-render of a unit binds the
   same arguments as its entry); no plane is exempt. */
int rx_canonical_like(rx_ctx* ctx, const rx_node* sinks, size_t n, rx_walk like, rx_walk* out);

int rx_digest(rx_ctx* ctx, rx_walk walk, char out65[65]);

/* Sets the context's last error text (a backend reports its own refusals through the same channel). */
int rx_set_error(rx_ctx* ctx, const char* msg, size_t n);

/* Builds a Loop and computes its boundary: the free operands are not given but found, as the
   maximal loop-invariant subexpressions of the body (reads under a dispatch arm are gated first).
   carries, inits and nexts are ncarries long; exit_cond, count and dynamic are NULL for none. */
int rx_new_loop(rx_ctx* ctx, rx_node index, const rx_node* carries, size_t ncarries,
                const rx_node* inits, const rx_node* nexts, int64_t start, int64_t stop,
                int64_t step, const rx_node* body_sinks, size_t nsinks,
                const rx_node* exit_cond /* NULL */, const rx_node* exit_values, size_t nexit,
                const rx_node* count /* NULL */, int count_tail, const char* dynamic /* NULL */,
                rx_node* out);

/* The roots with every node of the mapping (from[i] becomes to[i], by handle) replaced; an
   unchanged subtree is reused, not copied. */
int rx_substitute(rx_ctx* ctx, const rx_node* roots, size_t nroots, const rx_node* from,
                  const rx_node* to, size_t nmap, rx_node* buf, size_t cap, size_t* n);

/* The node over other operands, every other field kept (a loop is minted again by the loop
   constructor). */
int rx_rebuild(rx_ctx* ctx, rx_node node, const rx_node* operands, size_t nops, rx_node* out);

int rx_dump(rx_ctx* ctx, const rx_node* roots, size_t n, char* buf, size_t cap, size_t* len);

int rx_load(rx_ctx* ctx, const char* text, size_t n, rx_node* roots, size_t cap, size_t* nroots);

#ifdef __cplusplus
}
#endif
#endif
