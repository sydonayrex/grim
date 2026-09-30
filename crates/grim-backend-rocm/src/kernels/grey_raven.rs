//! GreyRaven (WS-E) probe kernel for the sparse FP8 SWMMAC layout.
//!
//! E6 needs three facts that the intrinsic's type signature does not give:
//! the lane-to-element mapping of the A and B fragments, the C/D fragment
//! mapping, and how the `u32` sparsity index corresponds to a 2:4 pattern. This
//! kernel is built so none of them has to be guessed *into* the kernel.
//!
//! The trick is that the A and B operands arrive as **raw byte buffers** and are
//! forwarded into the MMA's VGPRs unchanged. No fragment-packing helper is
//! applied, so whatever layout the hardware wants can be supplied from the
//! host and tested. The kernel only supplies the parts that are genuinely
//! fixed: the sparsity index operand, and writing every C lane out.
//!
//! That turns layout discovery into a host-side search: fill A and B with a
//! known pattern (the identity), run, and see which packing makes C come out as
//! the identity. A kernel that assumed a mapping would return finite plausible
//! garbage for every wrong guess, which is the failure mode this file exists to
//! avoid.
//!
//! The tile is 16x16x32 over a wave32, per the intrinsic name and S5's measured
//! signature:
//!   A : 16x32 FP8, 2:4 sparse  -> 256 B stored (512 B halved), 8 B/lane  = v2i32
//!   B : 32x16 FP8, dense       -> 512 B,                  16 B/lane = v4i32
//!   C : 16x16 f32              -> 1024 B,                 32 B/lane = v8f32
//!   sparsity index             -> u32, one per lane
//!
//! Guarded to gfx1200/gfx1201 with no `#else`: this is not a format that
//! degrades, it is an instruction that exists on exactly these parts.

/// HIP source for [`crate::kernels::grey_raven::PROBE_SOURCE`].
pub const PROBE_SOURCE: &str = r#"
#if (defined(__gfx1200__) || defined(__gfx1201__)) && defined(__HIP__)

typedef int   gr_v2i __attribute__((vector_size(8)));
typedef int   gr_v4i __attribute__((vector_size(16)));
typedef float gr_v8f __attribute__((vector_size(32)));

/// One 16x16x32 sparse FP8 SWMMAC with every operand supplied verbatim.
///
/// A is 8 bytes/lane, B is 16, C/D is 32, and `sidx` is the per-lane sparsity
/// index. Nothing is packed or permuted here on purpose: the host supplies
/// bytes and the host also checks the result, so a layout hypothesis that is
/// wrong shows up as a mismatch instead of being baked into the kernel.
///
/// Writes 8 floats per lane to `C_out`, which is 32 lanes * 8 * 4 = 1024 B.
extern "C" __global__ void grim_grey_raven_probe(
    const gr_v2i* __restrict__ A,     // 256 B  (8 B/lane)
    const gr_v4i* __restrict__ B,     // 512 B  (16 B/lane)
    unsigned int  sidx,               // per-lane sparsity index for A
    float*        __restrict__ C_out) // 1024 B (32 B/lane)
{
    const int lane = threadIdx.x;
    const gr_v2i a = A[lane];
    const gr_v4i b = B[lane];
    // D starts at zero rather than an incoming accumulator: this probe measures
    // the product, and a non-zero D would make the lane->element mapping of the
    // result ambiguous between "A x B" and "D + A x B".
    const gr_v8f d = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
        a, b, (gr_v8f){0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, sidx);
    C_out[lane * 8 + 0] = d[0];
    C_out[lane * 8 + 1] = d[1];
    C_out[lane * 8 + 2] = d[2];
    C_out[lane * 8 + 3] = d[3];
    C_out[lane * 8 + 4] = d[4];
    C_out[lane * 8 + 5] = d[5];
    C_out[lane * 8 + 6] = d[6];
    C_out[lane * 8 + 7] = d[7];
}
#endif
"#;

/// Same as [`PROBE_SOURCE`]'s kernel, but the sparsity index is read per lane.
///
/// The ISA is explicit that the index is per lane: "each lane has 8 index values per
/// lane", and group c's word for row r lives in lane (c>>2)*16 + r. A scalar index
/// therefore cannot express a real 2:4 pattern -- it would apply one lane's index to
/// all 32, and every row would collapse onto the same group selection. The scalar
/// form is kept because the layout probes that use it are about A and B, but the
/// end-to-end product test needs the real per-lane form.
pub const PROBE_LANE_IDX_SOURCE: &str = r#"
#if (defined(__gfx1200__) || defined(__gfx1201__)) && defined(__HIP__)
typedef int   gr_v2i __attribute__((vector_size(8)));
typedef int   gr_v4i __attribute__((vector_size(16)));
typedef float gr_v8f __attribute__((vector_size(32)));
extern "C" __global__ void grim_grey_raven_probe_lane_idx(
    const gr_v2i* __restrict__ A,
    const gr_v4i* __restrict__ B,
    const unsigned int* __restrict__ sidx,   // one u32 per lane
    float* __restrict__ C_out)
{
    const int lane = threadIdx.x;
    const gr_v2i a = A[lane];
    const gr_v4i b = B[lane];
    const unsigned int s = sidx[lane];
    const gr_v8f d = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(
        a, b, (gr_v8f){0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f}, s);
    C_out[lane * 8 + 0] = d[0]; C_out[lane * 8 + 1] = d[1];
    C_out[lane * 8 + 2] = d[2]; C_out[lane * 8 + 3] = d[3];
    C_out[lane * 8 + 4] = d[4]; C_out[lane * 8 + 5] = d[5];
    C_out[lane * 8 + 6] = d[6]; C_out[lane * 8 + 7] = d[7];
}
#endif
"#;
