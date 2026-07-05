//! Dashboard — composes the panels into one top-level VStack tree.

use super::panels::{build_header, build_job_history_panel, build_rocm_panel, build_training_panel};
use super::view_kind::{View, ViewKind};
use crate::view_model::ViewModel;

/// Build the dashboard tree. Returns a `View` tree with a header
/// row plus a three-column main row (ROCm / Hyperparameters / Job History).
pub fn build_dashboard(vm: &ViewModel) -> View {
    let header = build_header(vm);
    let rocm = build_rocm_panel(vm);
    let training = build_training_panel(vm);
    let jobs = build_job_history_panel(vm);

    let main_row = View::with_debug(ViewKind::HStack, "main_row")
        .child(rocm)
        .child(training)
        .child(jobs);

    View::with_debug(ViewKind::VStack, "dashboard").child(header).child(main_row)
}
