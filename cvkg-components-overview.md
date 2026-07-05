# CVKG Components Overview

## Widget Constructor Patterns

CVKG Components follows a **builder pattern** for most widgets, but some use **constructor + methods**. Here are the patterns:

### Pattern 1: Constructor + Builder Methods (Most Common)

```rust
// Button - label and callback required, then chain modifiers
Button::new("Click me", || println!("clicked"))
    .variant(ButtonVariant::Default)    // optional
    .size(ButtonSize::Large)             // optional
    .disabled(false);                    // optional
```

### Pattern 2: Constructor with Multiple Required Params

```rust
// Input - label, initial value, and callback all required
Input::new("Username:", "placeholder", |text| {
    // handle change
});
```

### Pattern 3: Range-based Widgets

```rust
// Slider - value, range, callback
Slider::new(0.5, 0.0..=1.0, |value| { /* handle */ });

// Stepper - label, initial value, callback
Stepper::new("Count", 0, |value| { /* handle */ });
```

### Pattern 4: Toggle-style Widgets

```rust
// Toggle - label, initial state, callback
Toggle::new("Enable feature", true, |is_on| { /* handle */ });
```

## ViewModel Integration Guide

The user's feedback indicates they pivoted to a **ViewModel pattern** for cleaner state management. Here's how to integrate:

### Step 1: Define Your ViewModel

```rust
#[derive(Debug, Clone)]
pub struct DashboardViewModel {
    // All UI text and state in one place
    pub button_label: String,
    pub is_loading: bool,
    pub slider_value: f32,
    pub toggle_states: [bool; 3],
    pub input_text: String,
    pub selected_date: Date,
}

impl Default for DashboardViewModel {
    fn default() -> Self {
        Self {
            button_label: "Click me".to_string(),
            is_loading: false,
            slider_value: 0.5,
            toggle_states: [false, true, false],
            input_text: "Enter text...".to_string(),
            selected_date: Date { year: 2026, month: 6, day: 27 },
        }
    }
}
```

### Step 2: Create UI State Actions

```rust
impl DashboardViewModel {
    pub fn on_button_click(&mut self) {
        self.is_loading = true;
        // simulate async work
        self.is_loading = false;
    }
    
    pub fn on_slider_change(&mut self, value: f32) {
        self.slider_value = value;
    }
    
    pub fn on_toggle_change(&mut self, index: usize, value: bool) {
        self.toggle_states[index] = value;
    }
    
    pub fn on_input_change(&mut self, text: String) {
        self.input_text = text;
    }
}
```

### Step 3: Build Views from ViewModel

```rust
pub fn build_dashboard(vm: &DashboardViewModel, vm_clone: Arc<Mutex<DashboardViewModel>>) -> VStack {
    VStack::new(16.0)
        .child(
            Button::new(&vm.button_label, move || {
                let mut vm = vm_clone.lock().unwrap();
                vm.on_button_click();
            })
            .loading(vm.is_loading)
        )
        .child(
            Slider::new(vm.slider_value, 0.0..=1.0, move |v| {
                let mut vm = vm_clone.lock().unwrap();
                vm.on_slider_change(v);
            })
        )
        .child(
            VStack::new(8.0)
                .child(Toggle::new("Feature 1", vm.toggle_states[0], move |v| {
                    let mut vm = vm_clone.lock().unwrap();
                    vm.on_toggle_change(0, v);
                }))
                .child(Toggle::new("Feature 2", vm.toggle_states[1], move |v| {
                    let mut vm = vm_clone.lock().unwrap();
                    vm.on_toggle_change(1, v);
                }))
        )
}
```

## Dashboard Example: Step-by-Step Construction

Here's a complete dashboard example showing the widget constructor patterns:

```rust
use cvkg_components::*;
use cvkg_core::{Rect, Renderer, View};
use std::sync::{Arc, Mutex};

// 1. Define the ViewModel
#[derive(Debug, Clone)]
pub struct DashboardState {
    pub username: String,
    pub notifications: usize,
    pub volume: f32,
    pub dark_mode: bool,
    pub selected_tab: usize,
    pub search_query: String,
}

impl Default for DashboardState {
    fn default() -> Self {
        Self {
            username: "Viking".to_string(),
            notifications: 3,
            volume: 0.75,
            dark_mode: false,
            selected_tab: 0,
            search_query: String::new(),
        }
    }
}

// 2. Build the Dashboard View
pub struct Dashboard {
    state: Arc<Mutex<DashboardState>>,
}

impl Dashboard {
    pub fn new() -> Self {
        Self {
            state: Arc::new(Mutex::new(DashboardState::default())),
        }
    }
    
    pub fn build_view(&self) -> VStack {
        let state = self.state.clone();
        let vm = state.lock().unwrap().clone();
        
        VStack::new(12.0)
            .child(self.build_header(&vm))
            .child(self.build_stats_row(&vm))
            .child(self.build_controls(&vm))
            .child(self.build_search(&vm))
    }
    
    fn build_header(&self, vm: &DashboardState) -> HStack {
        let state = self.state.clone();
        HStack::new(16.0)
            .child(Text::new(format!("Welcome, {}!", vm.username))
                .font_size(24.0)
                .color(theme::accent()))
            .child(NotificationBadge::new(vm.notifications, move || {
                let mut s = state.lock().unwrap();
                s.notifications = 0;
            }))
    }
    
    fn build_stats_row(&self, vm: &DashboardState) -> HStack {
        let state = self.state.clone();
        HStack::new(16.0)
            .child(Card::new(
                VStack::new(8.0)
                    .child(Text::new("Volume").font_size(12.0).color(theme::text_muted()))
                    .child(Text::new(format!("{:.0}%", vm.volume * 100.0))
                        .font_size(20.0).color(theme::text())),
            ))
            .child(Card::new(
                VStack::new(8.0)
                    .child(Text::new("Mode").font_size(12.0).color(theme::text_muted()))
                    .child(Toggle::new("", vm.dark_mode, move |v| {
                        let mut s = state.lock().unwrap();
                        s.dark_mode = v;
                    })),
            ))
    }
    
    fn build_controls(&self, vm: &DashboardState) -> VStack {
        let state = self.state.clone();
        VStack::new(8.0)
            .child(Slider::new(vm.volume, 0.0..=1.0, move |v| {
                let mut s = state.lock().unwrap();
                s.volume = v;
            }))
    }
    
    fn build_search(&self, vm: &DashboardState) -> HStack {
        let state = self.state.clone();
        HStack::new(16.0)
            .child(
                Input::new("Search...", vm.search_query.as_str(), move |text| {
                    let mut s = state.lock().unwrap();
                    s.search_query = text;
                })
                .frame(Some(300.0), Some(40.0))
            )
    }
}

// 3. Implement View trait
impl View for Dashboard {
    type Body = VStack;
    
    fn body(self) -> Self::Body {
        self.build_view()
    }
    
    fn render(&self, renderer: &mut dyn Renderer, rect: Rect) {
        let vm = self.state.lock().unwrap().clone();
        let view = self.build_view();
        view.render(renderer, rect);
    }
}
```

## Common Patterns Summary

| Widget | Constructor Pattern | Key Methods |
|--------|---------------------|-------------|
| Button | `Button::new(label, callback)` | `.variant()`, `.size()`, `.disabled()`, `.loading()` |
| Input | `Input::new(label, value, callback)` | `.frame()`, `.disabled()` |
| Toggle | `Toggle::new(label, is_on, callback)` | None |
| Slider | `Slider::new(value, range, callback)` | `.step()` |
| Stepper | `Stepper::new(label, value, callback)` | None |
| SecureField | `SecureField::new(placeholder, value, callback)` | None |
| DatePicker | `DatePicker::new(callback, initial_date)` | `.frame()` |
| Combobox | `Combobox::new(options, selected, callback)` | `.frame()` |

This approach keeps your UI state centralized in a ViewModel while making widget construction predictable.