# NPU KernelBench 通用数值与布局规则

本文件是**通用经验规则**，不针对任何具体算子。优先级低于 `definition.json` 和
`definition.reference`；冲突时以 reference 为准，并把冲突写进输出。

## 0. 通用

- 中间计算用 fp32，最后再 Cast 到输出 dtype；不要在 fp16/bf16 里做长累加。
- 设计前先看 `workload_summary`：dtype、rank/shape family、scalar 取值集合、
  tiny/huge、非末维、非法 workload，都要覆盖。
- 不要只实现“最后一维、连续、常见 dtype、常见参数”的主路径。
- 尾块、空 tensor、单元素、rank 变化、非连续 stride 都要有明确分支。
- 输出出现 inf/nan 时先判断：是输入溢出、reference 溢出，还是候选自身溢出。

## 1. 维度与数据搬运

- **末维**通常可以按连续块处理；**非末维**不能假设连续，必须按 stride 处理。
- 非末维不要用“one-element block”的 `DataCopyPad` 做 gather/scatter：
  `blockLen` 只有 1 个元素、靠 stride 跳行，既难对齐也容易触发不支持路径。
- 非末维优先：按完整行块搬入 UB，再在 UB 内按列 gather；或使用受支持的
  gather/copy 接口。保证每次 DMA 的 blockLen 是完整、对齐的字节数。
- 尾块用精确字节数的 `DataCopyPad`；不要用 raw `DataCopy` 碰非 32B 对齐的尾块。
- 同一份数据在 GM/UB 之间的搬入、搬出参数要对称；stride 单位是字节还是元素要统一。
- 为每个 shape family 检查：末维路径、非末维路径、极小 shape、极大 shape 是否都覆盖。

## 2. 归约与累加

- 累加器用 fp32；输出前 Cast。
- 大 shape 的 fp16 归约可能超出输出 dtype 范围；先确认 reference 是否也 inf。
  如果两边都 inf，这是数值边界，不要盲目改累加顺序。
- 归约维度不是最后一维时，注意 stride 和跨行访问；不要假设连续。
- `keepdim`、全归约、负 dim 都要处理。
- 前缀/扫描类操作有顺序依赖：分块后必须跨块传递前缀，不能各块独立计算。
- 归约顺序影响舍入；比较看 `max_abs/max_rel` 是否在容差内，不要强求逐位相等。

## 3. 排序 / 选择 / 掩码

- 明确升序/降序、取最大/取最小、`k` 的边界（0、等于维度、超过维度）。
- 非末维的排序/选择要走 stride 路径，不能复用末维连续路径。
- 相等值的 tie 规则按 reference 决定；如果 reference 只返回值，值相等即可。
- 输入可能含 NaN；NaN 的排序位置按 reference 处理，不要自行过滤。
- 用于填充/掩码的 sentinel（如 ±inf、-1）**绝不能**被选进最终结果或被写出；
  写回前检查 sentinel 是否残留在有效区间。
- 分块归并时，块边界、块内偏移、尾部块长度都要单独验证。

## 4. 索引 / 散射 / 原子

- 先确认 reference 是覆盖还是累加；重复索引的语义按 reference。
- 索引必须落在 `[0, size(dim))`；越界 workload 应报 `INVALID_WORKLOAD`，
  不要为它写 clamp/忽略逻辑。
- 索引 shape 可能小于源 shape（部分索引），按广播规则处理。
- 并行分块时，多个块可能写同一输出位置；需要原子累加或按输出归属划分，
  否则会丢更新。
- 写回前检查地址边界、元素/字节单位、负索引。

## 5. 归一化与反向

- 先写清楚 reference 的公式：mean/var 的位置、eps、weight/bias 顺序、分组方式。
- 反向要同时算 grad_input、grad_weight、grad_bias；缺一个就会 mismatch。
- 外部传入的 mean/std 不要自己重算。
- 统计量和平方和用 fp32；fp16 中间量容易 inf/nan。
- 参数（eps、scale、group 数）要按 workload 分支，不要写死。

## 6. 布局 / 缓存 / 步长

- 同一语义可能有多种布局（例如 interleaved 与 half-split、`[B,H,S,D]` 与
  `[B,S,H,D]`）；先看 reference 的取数方式和 stride，不能默认一种。
- 位置/索引类参数决定写入位置；注意越界、重复位置、部分更新。
- cos/sin 等辅助张量的 shape 可能是 `[B,S,D]` 或 `[B,1,S,D]`，注意 broadcast。
- 反向要区分“缓存梯度”和“状态梯度”的来源与形状。

## 7. 同步与事件

- 事件 ID 的获取、复用、释放规则见 `ascendc-event-sync` skill。
- 不要把 `PipeBarrier<PIPE_ALL>` 当作跨 pipe 同步的替代。
- 跨核 flag 必须所有 block 都执行，不能放在只有部分 block 进入的分支里。

## 8. 失败时先看什么

1. `evaluation.failing_workloads`：index、shape/dtype/scalar、correctness、error；
2. `evaluation.reference_invalid`：为 true 时不要改候选，reference 本身失败；
3. `definition.reference`：逐行对照公式、dtype、shape、边界；
4. 只改与失败 case 相关的分支，改完先跑失败 case + 原通过 case 回归。
