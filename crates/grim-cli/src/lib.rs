//! grim-cli library root — exposes training, config, eval, template registry, echo, and tui modules.
#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::unnecessary_cast,
    clippy::unnecessary_map_or,
    clippy::useless_format,
    clippy::redundant_closure,
    clippy::print_literal,
    clippy::field_reassign_with_default,
    clippy::unnecessary_sort_by,
    clippy::manual_repeat_n,
    clippy::if_same_then_else,
    clippy::manual_checked_ops,
    clippy::while_let_loop,
    clippy::new_without_default,
    clippy::let_unit_value,
    clippy::should_implement_trait,
    clippy::needless_range_loop,
    clippy::manual_ignore_case_cmp
)]

pub mod config;
pub mod doctor;
pub mod echo;
pub mod eval;
pub mod recipe;
pub mod template_registry;
pub mod train;
pub mod tui;
pub mod tune;
