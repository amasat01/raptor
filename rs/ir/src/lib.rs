//! The IR core: a per-context arena of node records reached through a C door (include/raptor_ir.h).
pub mod value;
pub mod class;
pub mod record;
pub mod door;
pub mod canon;
pub mod loops;
pub mod dialect;
pub mod ctor;
pub mod dump;
pub mod pseudo;

pub use class::{Class, Table};
pub use record::{Ctx, Fault, Node};
pub use value::Val;
