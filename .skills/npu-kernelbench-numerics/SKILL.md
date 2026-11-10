# NPU KernelBench 数值语义与边界规则

本文件是**经验规则**，不是任务契约。优先级低于 `definition.json` 和
`definition.reference`；当两者冲突时，以 reference 为准，并把冲突写进输出。

## 0. 通用

- 计算用 fp32 累加/中间量，最后再 Cast 到输出 dtype；不要在 fp16/bf16 里做长累加。
- 每个 workload 都要按 `shape`、`dtype`、`scalar` 参数分支设计；不要只实现“最后一维、
  连续、常见 dtype、常见参数”的主路径。
- 输入里可能同时有 fp16/bf16/fp32；同一个 kernel 要覆盖 `definition.inputs` 声明的所有 dtype。
- 出现 `inf`/`nan` 时先判断是输入溢出、reference 溢出，还是候选自身溢出；不要默认是 kernel 错。
- 尾块、空 tensor、单元素、rank 变化、非连续 stride 都要有明确分支。

## 1. 归约类（Sum / Mean / Cumsum / Norm 的统计量）

- 累加器用 fp32；输出前 Cast。
- 大 shape（numel ≥ 1e8）的 fp16 归约可能溢出到 inf：先看 reference 是否也 inf；
  如果 reference 也是 inf，这是 workload 边界，不是候选 bug。
- 归约维度不是最后一维时，注意 stride 和 cache 友好性；不能假设连续。
- `keepdim`、`dim=None`（全归约）、负 dim 都要处理。
- Cumsum/前缀和是**顺序依赖**：分块后需要跨块传递前缀，不能各块独立求和。
- 归约顺序会影响 fp 舍入；与 reference 比较时优先看 `max_abs/max_rel` 是否在容差内，
  不要强求逐位相等。

## 2. TopK

- 必须同时支持 `largest=true` 和 `largest=false`（找最小）。
- `dim` 可以是最后一维，也可以是 `dim=0` 或其他非末维；非末维时按 stride 取数。
- `k` 必须满足 `0 <= k <= x.shape[dim]`；`k=0`、`k=shape` 都要能跑。
- 一维超长输入（百万级）要走分块 + 归并，不能整段排序。
- bf16/fp16 的 NaN 语义：先确认 reference 对 NaN 的处理（通常 NaN 会排在最前/最后），
  不要自行过滤 NaN。
- reference 只返回 values（`torch.topk(...)[0]`）时，只比较 values；相等值的顺序不影响结果，
  但**值本身**必须一致。

## 3. Scatter / IndexPut / 索引类

- `torch.scatter_add` 是**累加**语义（重复 index 累加），不是覆盖。
- `index` 的取值必须落在 `[0, x.shape[dim])`；生成器如果给出越界 range，属于**非法 workload**，
  不要为它写 clamp/忽略逻辑，应在输出里报告 `INVALID_WORKLOAD`。
- `index` 的 shape 可能小于 `x`/`src`（部分索引），按广播规则处理。
- 重复 index 在并行分块时会跨核写同一输出；需要原子累加或按输出行归属划分，
  否则会丢更新。
- `accumulate=False` 时是覆盖语义；按 reference 判断，不要混用。

## 4. Norm 反向（AdaptiveInstanceNorm2D / GroupNormSwish / LayerNorm / RMSNorm）

- 先写清楚 reference 的公式：`mean`/`var`/`eps` 的位置、`weight`/`bias` 的顺序、
  `num_groups` 的划分。
- 反向要同时算 `grad_input`、`grad_weight`、`grad_bias`；缺一个就会 mismatch。
- `mean`/`std` 可能是外部传入的（AdaptiveInstanceNorm），不要自己重算。
- `swish_scale`、`eps`、`group` 边界都要按参数分支。
- fp16 中间量容易 inf/nan；统计量和平方和用 fp32。

## 5. Rope / KV Cache

- Rope 有 **interleaved** 和 **half-split** 两种布局；先看 reference 的 `cos/sin` shape 和
  取数方式，不能默认一种。
- `cos/sin` 的 shape 可能是 `[B, S, D]` 或 `[B, 1, S, D]`，注意 broadcast。
- `cache_position` 决定写入 cache 的位置；注意越界、重复位置、部分更新。
- KV cache 的 layout 可能是 `[B, H, S, D]` 或 `[B, S, H, D]`；按 reference 的 stride 处理。
- 反向要区分 `grad_key_cache`/`grad_value_cache` 与 `key_states`/`value_states` 的来源。

## 6. MoE

- routing scores / top-k / mask 的语义以 reference 为准；注意 `drop_pad_mode`、
  `expert` 数、`scales` 的 shape。
- 排序/散射类操作要注意重复 token、空 expert、越界 expert id。
- 归一化（softmax/sum）的分母是否包含被 mask 的项，按 reference 判断。

## 7. 失败时先看什么

1. `evaluate_operator.failing_workloads`：index、shape/dtype/scalar、correctness、error；
2. `evaluate_operator.reference_invalid`：为 true 时不要改候选，reference 本身失败；
3. `definition.reference`：逐行对照公式、dtype、shape、边界；
4. 只改与失败 case 相关的分支，改完先跑失败 case + 原通过 case 回归。
