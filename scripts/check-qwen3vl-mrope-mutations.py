import subprocess, pathlib

M = pathlib.Path("crates/grim-models/vision/src/qwen3vl_clip/mrope.rs")
F = pathlib.Path("crates/grim-models/vision/src/qwen3vl_clip/forward.rs")
T = pathlib.Path("crates/grim-models/vision/tests/qwen3vl_mrope.rs")
FT = pathlib.Path("crates/grim-models/vision/tests/qwen3vl_clip_forward.rs")
om, of, ot, oft = M.read_text(), F.read_text(), T.read_text(), FT.read_text()

# Each mutation targets a rule the reference pins down and that is INVISIBLE at
# position 0, which is precisely why each needs a test that kills it.
muts = [
    ("half-split -> interleaved pairing", M,
     "let (x0, x1) = (x[pair], x[pair + half]);",
     "let (x0, x1) = (x[pair * 2], x[pair * 2 + 1]);"),
    ("half-split -> wrong partner offset", M,
     "out[pair + half] = x0 * sin + x1 * cos;",
     "out[pair + half] = x1 * sin + x0 * cos;"),
    ("rotation sign flipped", M,
     "out[pair] = x0 * cos - x1 * sin;",
     "out[pair] = x0 * cos + x1 * sin;"),
    ("theta reset at section removed", M,
     "            if first_of_section {",
     "            if false {"),
    ("theta_scale exponent 2/n -> n", M,
     "self.freq_base.powf(-2.0 / n as f32)",
     "self.freq_base.powf(-1.0 / n as f32)"),
    ("theta_scale applied every pair, not per section", M,
     "thetas[sec] *= theta_scale;",
     "thetas[(pair + 1) % 4] *= theta_scale;"),
    ("freq_base 10000 -> text tower's 10000000", F,
     "const ROPE_FREQ_BASE: f32 = 10000.0;",
     "const ROPE_FREQ_BASE: f32 = 10000000.0;"),
    ("position id per PATCH -> per merged block", M,
     "        for y in 0..py {\n            for x in 0..px {",
     "        for y in (0..py).step_by(merge) {\n            for x in (0..px).step_by(merge) {"),
    ("position id component order y,x -> x,y", M,
     "ids.push([y as i64, x as i64, y as i64, x as i64]);",
     "ids.push([x as i64, y as i64, x as i64, y as i64]);"),
    ("is_valid() never refuses a narrow head", F,
     "        if !params.is_valid() {",
     "        if false {"),
    ("test: identity assertion loosened", T,
     "        assert_eq!(\n            *a, *b,",
     "        assert_eq!(((*a - *b).abs() < 0.5), true);"),
    ("test: non-zero rotation guard removed", T,
     "        x.iter().zip(moved.iter()).any(|(a, b)| (a - b).abs() > 1e-6),",
     "        false,"),
]

# SCOPE OF THIS GATE
#
# It measures one question: does a WRONG rope implementation get caught? Those are
# the CODE mutations below, and they are what coverage of this op has to mean - a
# RoPE is the identity at position 0 and preserves norms under any orthogonal mix,
# so the only meaningful failures are behavioural.
#
# Two mutation classes are deliberately EXCLUDED, because a survivor from either
# says nothing about coverage:
#   * disabling a test (adding #[ignore]) cannot fail a suite;
#   * LOOSENING an assertion on a CORRECT implementation cannot fail either.
# The latter is worth stating plainly: "test assertion loosened" is not a real
# mutation here, and reporting it as a survivor would be noise dressed as rigor.
killed = survived = skipped = 0
for label, path, a, b in muts:
    orig = path.read_text()
    if a not in orig:
        print("SKIP     " + label)
        skipped += 1
        continue
    path.write_text(orig.replace(a, b, 1))
    r = subprocess.run(
        ["cargo", "test", "-p", "grim-models-vision",
         "--test", "qwen3vl_mrope", "--test", "qwen3vl_clip_forward"],
        capture_output=True, text=True, timeout=900)
    dead = "FAILED" in r.stdout or r.returncode != 0
    print(("KILLED   " if dead else "SURVIVED ") + label)
    killed += dead
    survived += (not dead)
    path.write_text(orig)

assert (M.read_text() == om and F.read_text() == of and
        T.read_text() == ot and FT.read_text() == oft), "restore mismatch"
print(f"\nkill rate: {killed}/{killed + survived}  (survivors {survived}, skipped {skipped})")
print("files restored")