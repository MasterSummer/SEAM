# Agent failure analysis and next loop

Date: 2026-11-25

## 1. Current result

```
L1+L2: 63 tasks
pass: 34
strict: 34/63 = 54.0%
valid: 34/54 = 63.0%
```

Failure attribution:

```
candidate     16
build          4
environment     6
invalid         3
```

Run exits:

```
0:   35
124: 27
1:    1
```

API is no longer the main issue. Timeout/build/environment/hard candidate are.

## 2. Weaknesses

### 2.1 Finalization

Examples:

```
NMS        last rewrite -> 31 COMPILE_ERROR
FusedAdd   last rewrite -> 50 COMPILE_ERROR
KvCache    COMPILE_ERROR
Gather     COMPILE_ERROR
```

The agent tends to make a "definitive fix" after a working candidate and then
submits without a final compile/source check.

### 2.2 Stopping / progress

27/63 tasks hit the 3h timeout. CTCLoss is the clearest case:

```
agent_exit=124
28/100
15 INCORRECT_SHAPE
48 RUNTIME_ERROR
9 INCORRECT_NUMERICAL
no local smoke
```

It was still estimating/optimizing when the budget ended. It did not freeze
shape/runtime first.

### 2.3 Runtime diagnosis

Many failures are runtime, not numerical:

```
NLLLoss 50 runtime
Dequant 49 runtime
Swiglu 51 runtime
Rotary 24 runtime
CTCLoss 48 runtime
```

The agent is much better at writing a formula than at inferring:

```
workspace / blockDim / UB / queue / shape dispatch / launch args
```

### 2.4 Verification

```
CTCLoss: no local smoke
NMS: local passed, final build broken
FusedRope: source drifted from solution.json
```

Verification is not a mandatory gate before final.

### 2.5 Environment

Six environment failures were treated too long as candidate work:

```
Embedding
DynamicQuant
GroupNormSwish
Sort
Split
TopKTopP
```

The agent should probe the card and switch before rewriting code.

### 2.6 State / checkpoint

v4 CTCLoss had empty notes/experiments and no local records. v1 had a complete
kernel and verified notes. The agent does not reliably preserve a recoverable
state.

### 2.7 Ambiguous spec

IndexPut duplicate + accumulate=False is undefined by the reference. The agent
can spend time chasing a non-existent deterministic target.

## 3. What "shape 收口" means

Not just the output tensor shape. It means:

1. rank / dim normalization, including negative dim;
2. scalar attr vs tensor input;
3. output shape, dtype, layout;
4. per-dimension mapping from logical index to physical offset;
5. 2D/3D/4D dispatch;
6. empty tensor, tail, non-last dim;
7. host tiling descriptor and kernel grid/block mapping;
8. broadcast rules and keepdim;
9. reference shape equality before any numerical comparison.

A task has shape closed only when:

```
tiny shape correct
+ largest shape correct
+ tail/non-last correct
+ output shape/dtype equal to reference
```

## 4. What "runtime 收口" means

Eliminate runtime errors before looking at numerical error:

1. workspace size for the largest case;
2. blockDim <= physical cores and no zero block;
3. UB/queue buffer sizes;
4. 32B alignment and padded windows;
5. tail masking;
6. dtype cast / round mode;
7. launch argument order;
8. multi-output pointer order;
9. no device/stream corruption from the previous task.

Runtime closed means:

```
build ok
+ tiny and largest run without RUNTIME_ERROR
+ no environment failure
```

Only then should numerical debugging start.

## 5. Failure modification protocol

1. Classify first: build / shape / runtime / numerical / environment.
2. Take the first error, not the cascaded errors.
3. Reproduce one UUID on a clean card.
4. Run the same UUID on a second card:
   - fails both -> candidate;
   - one passes -> environment.
5. For candidate runtime, inspect workspace/blockDim/UB/shape dispatch.
6. For shape, compare output shape and per-dim mapping on a tiny case.
7. For numerical, compare one element and one reduction.
8. Change one thing, re-run the same UUID, checkpoint only after it passes.

## 6. Required mechanisms

### 6.1 Checkpoint

After each verified step:

```
checkpoints/0001-tiny-pass
checkpoints/0002-runtime-pass
checkpoints/0003-matrix-pass
```

Final must use the highest verified checkpoint, never a broken current source.

### 6.2 Compile gate

Before final:

```
build ok
source == solution.json
tiny passes
```

If not, restore the last build-ok checkpoint.

### 6.3 Final freeze

Last 15-20 minutes: no large rewrite. Only compile, tiny/tail verification, and
source sync.

### 6.4 Timeout triage

A separate triage agent decides:

```
continue
stop
retry_api
retry_clean_card
```

and writes `triage.json`.

### 6.5 Multi-agent split

```
triage agent       -> classify and decide
shape agent        -> rank/dim/output/tiling mapping
runtime agent      -> workspace/blockDim/UB/launch
numerical agent    -> one-case numerical diff
verifier agent     -> build/source/tiny/tail/final gate
```

They share the same workspace and checkpoint, not separate solutions.

## 7. Current triage prototype

`src/scripts/triage_agent.py` reads run/final/local/log/notes and writes:

```
workspace/triage.json
run.json: triage_decision
```

CTCLoss v4 result:

```
decision=continue
goal=stabilize runtime/shape before full
```

## 8. Next loop

1. Add checkpoint + compile gate + final freeze.
2. Consume triage decision in scheduler.
3. Add shape/runtime diagnostic prompts/subagents.
4. Validate on the 29 retried failures.
5. Only then use level3 for iteration.
