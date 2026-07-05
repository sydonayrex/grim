//! Top-level UI module — view_kind shim, panel compositors, and
//! the dashboard tree.
//!
//! The renderer host (`main.rs`) consumes the [`View`] tree produced by
//! [`dashboard::build_dashboard`] and turns it into a real wgpu surface
//! via cvkg-core's runtime. While the renderer wiring is still pending,
//! the test suite asserts on [`View::kind()`] and [`View::debug_string()`]
//! to lock in the structural shape.

pub mod dashboard;
pub mod panels;
pub mod view_kind;

pub use dashboard::build_dashboard;
pub use panels::{build_header, build_job_history_panel, build_rocm_panel, build_training_panel};
pub use view_kind::{View, ViewKind, WidgetInner, WidgetSlot};
