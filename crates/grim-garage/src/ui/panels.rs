//! Panel compositors — one function per panel that turns a slice of
//! `ViewModel` into a `View` tree mounted inside the dashboard.
//!
//! Each compositor constructs both:
//!  - a CVKG widget for the live render (so the runtime has it ready),
//!  - a `View` introspection wrapper (`kind()` + `debug_string()`) so the
//!    test suite can lock in the structural shape.
//!
//! One panel per file keeps `mod.rs` short.

use cvkg_components::{Badge, BadgeVariant, Button, Slider, Stepper, Text, Toggle};

use super::view_kind::{View, ViewKind, WidgetInner, WidgetSlot};
use crate::view_model::ViewModel;

// ----- header -----

/// Top-of-window brand + sub-status row.
pub fn build_header(vm: &ViewModel) -> View {
    let title = View::with_debug(ViewKind::Text, vm.layout.window_title.clone())
        .widget_of(WidgetSlot {
            id: "header_title".into(),
            kind: WidgetInner::Text(Text::new(vm.layout.window_title.clone())),
        });
    let subtitle = View::with_debug(
        ViewKind::Text,
        format!(
            "poller every 5s • {} models • {} datasets • {} devices",
            vm.models.len(),
            vm.datasets.len(),
            vm.rocm_devices.len()
        ),
    );

    let start = View::with_debug(ViewKind::Button, "Start Training".to_string())
        .widget_of(WidgetSlot {
            id: "start_training".into(),
            kind: WidgetInner::Button(Button::new("Start Training", || {})),
        });

    View::with_debug(ViewKind::HStack, "header")
        .child(title)
        .child(subtitle)
        .child(start)
}

// ----- ROCm panel -----

/// Device summary + 4 toggles.
pub fn build_rocm_panel(vm: &ViewModel) -> View {
    let mut children: Vec<View> = Vec::with_capacity(vm.rocm_toggles.toggles.len() + 1);
    children.push(View::with_debug(
        ViewKind::Text,
        vm.rocm_toggles.device_summary.clone(),
    ));
    for toggle in &vm.rocm_toggles.toggles {
        let label = toggle.label.clone();
        let id_prefix = format!("rocm_toggle_{}", toggle.id);
        // Note: cvkg 0.3.3 Toggle has no `.disabled()` setter — render with
        // the current `checked` value regardless. The panel above ends
        // with a "disabled" suffix when enforced by the runtime.
        let w = Toggle::new(label.clone(), toggle.checked, |_| {});
        children.push(
            View::with_debug(ViewKind::Toggle, label.clone()).widget_of(WidgetSlot {
                id: id_prefix,
                kind: WidgetInner::Toggle(w),
            }),
        );
    }
    View::with_debug(ViewKind::Card, vm.rocm_toggles.panel_title.clone())
        .children_of(children)
}

// ----- training panel -----

/// Title + help text + mode picker + (sometimes quant picker) + LoRA rank stepper + LR slider.
pub fn build_training_panel(vm: &ViewModel) -> View {
    let title_text = vm.training_panel.panel_title.clone();
    let help_text = vm.training_panel.help_text.clone();

    let mode_index = vm
        .training_panel
        .mode_options
        .iter()
        .position(|m| m == &vm.training_config.training_mode)
        .unwrap_or(0) as i32;
    let mode_picker = View::with_debug(ViewKind::Picker, "training_mode").widget_of(
        WidgetSlot {
            id: "training_mode".into(),
            kind: WidgetInner::Stepper(Stepper::new("Training mode", mode_index, |_| {})),
        },
    );

    let quant_picker = if vm.training_panel.show_quant_format_picker {
        View::with_debug(ViewKind::Picker, "quant_format")
    } else {
        View::new(ViewKind::Text) // empty placeholder keeps the column count stable
    };

    let rank = vm.training_config.lora_rank as i32;
    let rank_stepper = View::with_debug(ViewKind::Stepper, "lora_rank").widget_of(
        WidgetSlot {
            id: "lora_rank".into(),
            kind: WidgetInner::Stepper(Stepper::new("LoRA rank", rank, |_| {})),
        },
    );

    let lr = vm.training_config.learning_rate as f32;
    let lr_slider = View::with_debug(ViewKind::Slider, "learning_rate").widget_of(
        WidgetSlot {
            id: "learning_rate".into(),
            kind: WidgetInner::Slider(Slider::new(lr, 0.0..=1e-3_f32, |_| {})),
        },
    );

    View::with_debug(ViewKind::Card, title_text).children_of(vec![
        View::with_debug(ViewKind::Text, help_text),
        mode_picker,
        quant_picker,
        rank_stepper,
        lr_slider,
    ])
}

// ----- job history panel -----

/// One Card per job — with a badge.
pub fn build_job_history_panel(vm: &ViewModel) -> View {
    let mut children: Vec<View> = Vec::with_capacity(vm.jobs.len());
    for card in &vm.jobs {
        let badge_label = card.badge_label();
        let variant = match card.status.as_str() {
            "completed" => BadgeVariant::Success,
            "failed" => BadgeVariant::Destructive,
            "running" => BadgeVariant::Success, // CVKG 0.3.3 has no Info; Success = green
            _ => BadgeVariant::Default,
        };
        let badge = Badge::new(badge_label.clone()).variant(variant);
        children.push(
            View::with_debug(ViewKind::Card, format!("job {}", card.job_id)).children_of(vec![
                View::with_debug(ViewKind::Badge, badge_label).widget_of(WidgetSlot {
                    id: format!("badge_{}", card.job_id),
                    kind: WidgetInner::Badge(badge),
                }),
                View::with_debug(ViewKind::Text, card.subtitle()),
            ]),
        );
    }
    let title = format!("Job History ({} jobs)", vm.jobs.len());
    View::with_debug(ViewKind::Card, title).children_of(children)
}

// ----- extension trait on View -----

#[allow(dead_code)]
trait ViewWidgetExt {
    fn widget_of(self, slot: WidgetSlot) -> Self;
}
#[allow(dead_code)]
impl ViewWidgetExt for View {
    fn widget_of(mut self, slot: WidgetSlot) -> Self {
        self.widget = Some(slot);
        self
    }
}
