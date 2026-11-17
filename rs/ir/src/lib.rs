#![warn(missing_docs)]
#![warn(clippy::missing_docs_in_private_items)]
//! The IR core: a per-context arena of node records reached through a C door (include/raptor_ir.h).
pub mod build;
pub mod canon;
pub mod cells;
pub mod class;
pub mod consts;
pub mod ctor;
pub mod dialect;
pub mod door;
pub mod dump;
pub mod indexform;
pub mod looprules;
pub mod loops;
pub mod quantity;
pub mod query;
pub mod record;
pub mod structkeys;
pub mod typed;
pub mod typing;
pub mod value;

pub use class::{Class, Table};
pub use record::{Ctx, Fault, Node};
pub use value::Val;
