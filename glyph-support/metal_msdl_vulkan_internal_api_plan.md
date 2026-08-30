# Vulkan — internal API bridge plan

> Detailed plan for approaching the Vulkan side of glyph, from first principles through to an implementable Vulkan bridge covering both compute and render paths. Not a substitute for hands-on Vulkan engineering — carry a real Vulkan dev for the implementation phase.

**Status:** Draft plan — ready for Vulkan implementation pass  
**Created:** 2026-02-28  
**Change log:**
- 2026-02-28 — Initial plan from first-principles Vulkan overview and hand-engineered glyph Vulkan conceptualization.

---

## 1. Purpose & scope

### 1.1 What this document is

This is the Vulkan-side companion to the Metal side of glyph. It covers:

- A **first-principles Vulkan overview** that an ML engineer without deep Vulkan experience can use to reason about device selection, queue families, command buffers, synchronization, descriptor sets, pipelines, and shader modules.
- A **hand-engineered conceptualization** of a Vulkan glyph build: the device setup, descriptor/model, compute dispatch path, and render path, described at the level of Vulkan objects and CPU-side control flow — enough to drive an implementation.
- An **internal bridging API sketch**: a shared trait or dispatch abstraction that Vulkan backends can implement so that higher-level glyph code can target Vulkan without per-backend ifdefs everywhere.
- A **test plan** and **verification approach** that locks the Vulkan contract early with a small, public, extensible harness.

### 1.2 What this document is not

- Not a Vulkan driver implementation guide — it assumes access to a real Vulkan driver and ICD.
- Not a prescriptive Vulkan feature set — the API coverage model is explicit about what is required vs. optional.
- Not a substitute for profiling — the plan intentionally leaves performance work for a later pass.

### 1.3 Guiding principles

- **Correctness before performance.** A minimal, correct Vulkan bridge beats a fast broken one.
- **Explicit capability model.** Every Vulkan feature that glyph uses must be enumerated and required-or-optional classified.
- **Small, public, extensible test surface.** The first Vulkan test should lock a narrow contract and be easy to extend when new Vulkan backends appear.
- **Separation of concerns.** Device lifetime, command recording, synchronization, descriptor management, and shader compilation should be kept modular enough to reason about independently.

---

## 2. Vulkan backend baseline (what is already wired)

Before designing the bridge, enumerate what Vulkan support already exists in the project. For each existing piece, record:

- What Vulkan objects it creates or owns (instance, device, queue, command pool, descriptor pool, pipeline, shader module, buffer, image, fence, semaphores, etc.).
- What CPU-side ownership model it uses (who creates, who destroys, who records commands, who submits, who syncs).
- What shader path it uses (SPIR-V, runtime-generated code, precompiled, reflection-driven).
- What synchronization it implements (fences, semaphores, pipeline barriers, memory barriers, ` VK_..._BIT` flags, explicit layout transitions).
- What resource management it uses (buffer/image allocation strategy, alignment, device-local vs host-visible vs host-coherent, upload/download stubs).

For this plan, the key is to know:

- Whether there is already a Vulkan device/instance bootstrap.
- Whether there is already a compute pipeline path.
- Whether there is already a render pipeline path.
- Whether there is already a descriptor/set model, or whether descriptors are per-dispatch ad hoc.
- Whether there is already a command recording lifecycle (pool per frame, per batch, per dispatch).
- Whether there is already a synchronization model (in-flight frames, fence per submission, semaphore for queue family transfers, etc.).

The Vulkan bridge design should fit the existing baseline where it exists, and fill gaps where it does not.

---

## 3. Vulkan API surface that glyph needs

### 3.1 Compute side

At minimum, a Vulkan backend for glyph likely needs:

- Device creation with the queue families required for compute (and optionally graphics, if the render path is in scope).
- Command pool(s) and command buffer lifecycle suitable for batched compute dispatch.
- Shader modules for whatever compute kernels glyph runs (SPIR-V, ideally generated from a shared kernel source or from a stable SPIR-V-friendly representation).
- Descriptor sets / bindings for the tensors/buffers the compute kernels touch.
- Pipeline layout(s) matching the shader interface.
- Compute pipelines (or at least a path to create them).
- Buffer allocation with correct usage flags (`VK_BUFFER_USAGE_*`) for device-local storage, staging, and host-visible readback where needed.
- Memory allocation and binding with awareness of device-local vs host-visible vs host-coherent trade-offs.
- Explicit synchronization around dispatches when multiple dispatches or resource transitions touch overlapping ranges.
- Optional: pipeline modifiers / specialization constants for runtime parameters.

### 3.2 Render side (if in scope)

If glyph also has a Vulkan render path, the surface grows to include:

- Swapchain setup and present mode selection.
- Render passes, subpasses, and attachment descriptions.
- Graphics pipeline(s) for the relevant draw paths.
- Vertex/index buffer bindings if applicable.
- Framebuffer and image view setup.
- Render with-load, render-with-dispatch, or post-compute render patterns if compute and graphics are interleaved.
- Presentation and swapchain synchronization.
- Layout transitions for color/depth attachments.

If the render path is not in scope for the initial bridge, say so explicitly.

### 3.3 Shared abstractions

Across compute and render, glyph likely benefits from shared Vulkan abstractions for:

- Device handle ownership and lifetime.
- Queue selection and submission.
- Resource creation helpers (buffers, images, view helpers).
- Descriptor management (pool, set layout, set updates).
- Command recording and submission helpers.
- Synchronization helpers (fences, semaphores, barriers).
- Error handling and validation-layer-friendly behavior.
- Capability probing and fallback decisions.

---

## 4. Internal bridging API (VulkanBackend impl of a shared trait)

The core design move is: define an internal trait or dispatch abstraction that Vulkan backends implement, and that higher-level glyph code can target.

### 4.1 Why a trait/dispatch abstraction

- It keeps Vulkan-specific code from infecting higher-level glyph logic.
- It makes Vulkan one implementation among several, with a consistent contract.
- It makes tests easier: a Vulkan backend can be exercised through the same trait as other backends.
- It makes the capability model explicit: the trait can express what is required vs optional, and Vulkan implementations can advertise what they support.

### 4.2 Sketch of the abstraction

Without pre-empting the real API design, the abstraction should cover:

- Device/instance lifecycle.
- Resource creation (buffers/images) with a stable resource handle type.
- Descriptor/set/pipeline creation helpers, or a way to build them from a shared description.
- A dispatch/run interface for compute.
- A render interface if render is in scope.
- Synchronization primitives or helpers.
- Error reporting that higher-level code can map to glyph error model.
- Capability queries needed by higher-level code.

### 4.3 What VulkanBackend owns

A VulkanBackend impl should own the Vulkan-specific objects and present a stable internal API to the rest of glyph. At minimum, think through:

- Instance and device handles.
- Queue handles and family indices.
- Command pool(s) and command buffer allocation strategy.
- Descriptor pool and set layout policy.
- Pipeline cache policy.
- Resource allocation helpers and any allocator abstraction.
- Synchronization handle lifecycle.

### 4.4 What the rest of glyph should NOT see

Higher-level code should not be forced to know:

- Raw Vulkan handles except through the abstraction.
- Queue family indices except as mediated by the backend.
- SPIR-V binary construction unless that is part of the shared contract.
- Validation-layer-specific behavior except at the backend boundary.

---

## 5. Placeholder/fallback accounting (the eight functions)

The user explicitly asked to account for eight functions that are **placeholders/Java-class-and-C-function pairs** subject to replacement. Treat them as a first-class part of the plan.

### 5.1 What to record for each placeholder

For each of the eight functions, record:

- Where it lives today (class + C function pair).
- What it is supposed to do at a functional level.
- What Vulkan equivalent would plausibly replace it.
- Whether the replacement is compute, render, or both.
- Whether the replacement needs new Vulkan objects, new shaders, new synchronization, or new resource layouts.
- Whether the replacement depends on any other placeholder being implemented first.
- What the fallback behavior should be before replacement.
- What tests would validate the replacement.

### 5.2 Placeholder-to-Vulkan mapping discipline

For each placeholder, avoid vague “replace with Vulkan” language. Instead, write down the concrete Vulkan work items:

- Buffer/image creation and usage.
- Shader module requirements.
- Descriptor/binding requirements.
- Pipeline requirements.
- Command recording and submission requirements.
- Synchronization requirements.
- Error handling requirements.

This is what makes the plan implementable: each placeholder becomes a small, reviewable Vulkan change list rather than a hand-wavy replacement.

### 5.3 Fallback strategy for placeholders

Until a placeholder is replaced, the Vulkan path should have a defined fallback:

- Either emulate the intended behavior in terms of existing Vulkan paths, or
- Fail with a clear, traceable error that explains the missing Vulkan feature/work item.

Do not leave placeholders that silently do the wrong thing in Vulkan. Placeholder behavior should be intentional, documented, and testable.

---

## 6. Functions handled elsewhere // NOT in scope

Explicitly enumerate what is out of scope for this Vulkan bridge plan so the boundary is not accidentally expanded.

Likely “handled elsewhere” items include things like:

- General Vulkan boilerplate that already exists and does not need re-planned.
- Higher-level glyph logic that should be shared and not Vulkan-specific.
- Application-level integration that is orthogonal to the Vulkan backend.
- Performance tuning that should come after a correct Vulkan bridge exists.

The point is not to list everything imaginable, but to draw a line: if it is not part of the Vulkan backend bridge or the eight placeholders, put it elsewhere.

---

## 7. X-plane enumeration & fallback strategy

“X-plane” here means the set of Vulkan feature/capability planes that glyph may depend on and that must be enumerated before implementation so that fallback behavior is deliberate.

### 7.1 Planes to enumerate

Possible planes include, but are not limited to:

- Queue family support (compute-only, graphics-only, transfer, sparse, etc.).
- SPIR-V feature support needed by the shaders.
- Descriptor set layout limits and binding model constraints.
- Buffer usage and memory property requirements.
- Image usage and layout requirements if images are involved.
- Synchronization capabilities (fences, semaphores, barriers, queue family ownership transfer if relevant).
- Extension requirements (optional or mandatory).
- Device feature bits that affect correctness or fallback.

### 7.2 Fallback strategy per plane

For each plane, decide:

- Is it required for correctness?
- Is it required for performance but not correctness?
- Is it optional and can be absent with a degraded-but-correct path?
- What error or fallback occurs if the plane is absent?
- How is the absence detected (caps checking, extension presence, physical device limits)?

### 7.3 Why this matters

Without this enumeration, Vulkan implementation turns into a hunt for “why did this work on one device and not another?” The X-plane list is the device-compatibility contract for the Vulkan backend.

---

## 8. Test plan & coverage

### 8.1 Lock the contract first

The first Vulkan tests should lock the internal API contract, not try to cover every possible Vulkan path. Start with:

- Device creation and capability probing.
- A minimal compute dispatch through the Vulkan backend.
- Resource creation, descriptor binding, and dispatch return behavior.
- A defined fallback behavior for a missing/platform-specific feature.

### 8.2 Make the test surface public and extensible

The test harness should be:

- Public enough that other Vulkan backends can plug into it.
- Extensible so new functionality can add tests without rewriting the harness.
- Deterministic enough to be useful in CI, with clear pass criteria.

### 8.3 Coverage goals

Coverage should be tied to the API contract and the placeholder list:

- Each placeholder replacement should have at least one test that validates the Vulkan behavior intended for it.
- Each required Vulkan plane should have at least one test that exercises it or validates the fallback.
- Each shared abstraction in the VulkanBackend trait/interface should have at least one test that exercises it through a Vulkan implementation.

### 8.4 What to avoid

- Do not try to freeze a massive Vulkan test matrix early.
- Do not rely on one exotic device as the only Vulkan test surface if avoidable.
- Do not let Vulkan tests depend on external runtime behavior that cannot be reproduced.

---

## 9. Explicit non-goals (boundaries)

This plan is intentionally bounded. Non-goals include:

- Writing production Vulkan driver code or replacing an existing Vulkan stack unless the plan explicitly says so.
- Full performance optimization in the first pass.
- Supporting every possible Vulkan device variant before the bridge is correct.
- Defining the final public API of glyph if that is a separate decision.
- Replanning existing Vulkan code that already works and does not interact with the eight placeholders.

---

## 10. Open questions

The plan should end with open questions rather than pretend certainty. Likely open questions:

- Which Vulkan queue families are actually required by the current glyph Vulkan baseline?
- Which placeholders are compute-only, which are render, and which touch both?
- What SPIR-V/source path feeds the Vulkan compute kernels, and who maintains it?
- Are swapchain/render paths in scope for the initial Vulkan bridge, or only compute?
- What is the resource allocation model: device-local allocator, per-dispatch buffers, shared pools, or something else?
- What synchronization model is sufficient for the first Vulkan bridge?
- Which fallback behaviors are acceptable for the eight placeholders before replacement?
- What capability checks must be explicit vs implicit?

---

## 11. Sequencing & dependencies

A reasonable sequence:

1. Fix the Vulkan baseline inventory: what objects exist, what lifecycle exists, what shaders exist, what synchronization exists.
2. Write down the Vulkan API surface glyph needs from section 3, trimmed to what is actually in scope.
3. Define the shared trait/dispatch abstraction from section 4.
4. Enumerate the eight placeholders from section 5 and map each to concrete Vulkan work items.
5. Enumerate Xplanes and fallback rules from section 7.
6. Build the first Vulkan tests from section 8 — lock the contract.
7. Implement the Vulkan backend behind the abstraction.
8. Replace placeholders one by one with tests guarding each replacement.
9. Profile and refine later, once correctness is established.

---

## Appendix A — Coverage matrix (planning tool)

Use a matrix to track what is required vs optional across the Vulkan path. The point is not the exact shape of the matrix, but that it exists and is maintained.

For example, the matrix should at minimum help answer:

- Which functions are pure compute?
- Which functions require graphics/render support?
- Which functions are placeholders to be replaced?
- Which Vulkan planes each function depends on?
- Which functions are test-covered today?

The current plan should make it easy to see that functions like softmax, rms_norm, and rope are in scope for coverage/progress tracking — whether as compute paths, render paths, or both — and that each has a clear Vulkan dependency and fallback story.

---

## Appendix B — Vulkan caps helpers already exist

Use existing Vulkan capability helpers where possible instead of inventing new ones. The existing Vulkan caps infrastructure is the natural place to express:

- Device capability queries.
- Feature/extension presence checks.
- Limit checks that drive fallback.
- The device-compatibility contract for the Vulkan backend.

If the existing Vulkan caps helpers are incomplete for the eight placeholders or the new VulkanBackend trait, extend them rather than bypassing them. The Vulkan backend should get its capability answers from one disciplined source.

---

## Implementation notes (hand-engineered Vulkan glyph)

This appendix records the Vulkan conceptualization at the level needed to drive implementation, not to replace Vulkan expertise.

### Device setup

- Create instance with the extensions/validation needed for the target environment.
- Enumerate physical devices and choose one based on the required queue families and features.
- Create the device with the queue families needed for compute and, if in scope, graphics.
- Retrieve queue handles.
- Decide on command pool policy: per-queue, per-thread, per-frame, or per-batch, depending on submission patterns.

### Descriptor/model

- Decide whether descriptors are:
  - Per-dispatch ad hoc,
  - Pooled,
  - Or managed through a higher-level set-layout/pipeline-layout model.
- If glyph kernels share a stable binding interface, prefer a shared descriptor/set layout model so pipeline creation and descriptor updates are predictable.
- If the binding interface is dynamic, make the abstraction capable of representing that without leaking raw Vulkan details everywhere.

### Compute dispatch path

A minimal Vulkan compute dispatch path is:

1. Create the compute shader module(s).
2. Define pipeline layout and compute pipeline.
3. Create or bind buffers/images with appropriate usage.
4. Allocate and update descriptor sets.
5. Record a command buffer:
   - Bind pipeline,
   - Bind descriptor sets,
   - Optionally set any dynamic state or specialization constants,
   - Dispatch.
6. Submit with the synchronization model the backend uses.
7. Optionally wait or track completion via fence/semaphore.

The VulkanBackend abstraction should make this sequence expressible as a stable internal API, not as raw Vulkan calls scattered through glyph.

### Render path (if in scope)

If render is in scope, the Vulkan path adds:

- Swapchain and surface setup.
- Render pass and attachment descriptions.
- Framebuffer setup.
- Graphics pipeline(s) where needed.
- Command recording for render passes.
- Present and swapchain synchronization.

If render is deferred, mark it as deferred and list the Vulkan objects it will eventually need so the abstraction can accommodate it later.

### Synchronization sketch

At minimum, decide:

- How the Vulkan backend waits for prior work to complete.
- How it avoids overwriting resources still in use.
- Whether it uses fences per submission, in-flight resource slots, or another model.
- Whether queue-family transfers require semaphores or ownership-transfer barriers.

This synchronization model should be part of the VulkanBackend abstraction, because higher-level code will depend on it being well-defined.

---

## Test contract (public, extensible, first Vulkan tests)

The first Vulkan tests should be small and contract-locking. A workable first pass:

- **Device capability test:** confirms the Vulkan backend can advertise the required capabilities and detect the fallback conditions from the X-plane list.
- **Minimal compute dispatch test:** confirms the Vulkan backend can create/compile/bind/dispatch a compute kernel through the shared abstraction and return results.
- **Placeholder behavior test:** for each placeholder, a test that documents the current fallback behavior and can be extended when the Vulkan replacement lands.
- **Resource lifecycle test:** confirms buffer/image creation, usage, and destruction behave as expected through the Vulkan backend.
- **Render-path test, if in scope:** a minimal render submission that validates the graphics path through the same abstraction.

These tests should be written so that adding a new Vulkan backend or a new placeholder replacement only requires adding a small number of targeted cases, not rewriting the harness.

---

## Conclusion

The Vulkan side of glyph should be approached as:

1. Understand the existing Vulkan baseline.
2. Enumerate the Vulkan API surface glyph actually needs.
3. Define a VulkanBackend abstraction that isolates Vulkan from the rest of glyph.
4. Treat the eight placeholders as first-class work items with concrete Vulkan mappings and fallback behaviors.
5. Enumerate X-planes and fallback rules so device compatibility is explicit.
6. Lock the contract with a public, extensible Vulkan test harness.
7. Implement and replace placeholders one at a time, guarded by tests.
8. Defer performance work until correctness is established.

That sequence keeps the Vulkan effort concrete, reviewable, and testable, and avoids turning the Vulkan bridge into an unbounded Vulkan refactor.
