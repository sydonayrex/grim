#![allow(ambiguous_glob_reexports)]
#![allow(clashing_extern_declarations)]
#![allow(dead_code)]
#![allow(improper_ctypes_definitions)]
#![allow(suspicious_runtime_symbol_definitions)]
#![allow(unused_imports)]
// The build script selects exactly one committed bindings file for the
// installed HIP patch and copies (or, under `dynamic`, rewrites) it into
// OUT_DIR/bindings.rs.
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
