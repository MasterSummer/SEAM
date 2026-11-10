# Ascend C 同步与事件 ID 规则

本文件是**通用经验规则**，不针对任何具体算子。优先级低于任务契约和 reference。

## 0. 症状

- 极小 shape 也超时，且没有任何错误日志；
- 同一个候选反复超时，reference 能正常跑完；
- 有时只有第一个 workload 超时，后面全部 `NOT_RUN`。

这类现象优先怀疑**事件 ID 耗尽**或**SetFlag/WaitFlag 不配对**，而不是性能问题。

## 1. FetchEventID 的位置

- `FetchEventID` 应该在 `Init`/`Process` 开头**取一次**，保存为成员变量，循环里复用。
- **不要**在 per-tile、per-row、per-chunk、per-element 循环里反复 `FetchEventID`。
  事件 ID 池是有限的；取完不还，后续 `SetFlag/WaitFlag` 会永久等待。
- 如果确实需要在循环里取，必须在 `WaitFlag` 之后调用 `ReleaseEventID`，
  保证取还配对。
- 不要硬编码 `EVENT_ID0/1/...`，它们可能已被 TQue/TPipe 占用；用
  `FetchEventID` 获取属于当前 pipe 的 ID。

## 2. SetFlag / WaitFlag

- 每个 `SetFlag` 必须有且只有一个对应的 `WaitFlag`；同一 ID 不要嵌套使用。
- 不要把 `SetFlag` 放在只有部分分支会执行的路径里，否则另一条路径会永远等。
- `WaitFlag` 之前必须确认 `SetFlag` 已经执行；跨 pipe 的依赖要显式建 flag，
  不要依赖 `PipeBarrier<PIPE_ALL>` 代替。
- 循环里复用同一个 ID 时，确认上一轮的 `WaitFlag` 已经消费掉，再 `SetFlag`。

## 3. 跨核同步

- `SyncAll`、`CrossCoreSetFlag`/`CrossCoreWaitFlag` 必须所有 block 都执行；
  不要放在 `if (blockIdx == 0)` 或只有部分 core 进入的分支里。
- 如果某些 core 提前 `return`，其余 core 的跨核等待会死锁。
- 跨核 flag 的 event ID 也要按上面的规则取还，不要在循环里反复取。

## 4. 排查方法

1. 用**单个 workload** 跑（`--limit 1` / 只保留一个 workload）；
2. 设置 `ASCEND_LAUNCH_BLOCKING=1`，让错误在同步点暴露；
3. 在 `Init`、每个循环边界、每次 `SetFlag/WaitFlag` 前后加 host/kernel marker，
   看最后一个 marker 停在哪里；
4. 如果是事件 ID 耗尽，通常最后停在某个 `WaitFlag`，前面的 kernel 逻辑都正常；
5. 修复后先用极小 shape + 原失败 shape 各跑一次，再跑全量。

## 5. 常见错误模式

- 在 `for` 循环里 `FetchEventID`，但没有 `ReleaseEventID`；
- 同一个 event ID 在嵌套循环里重复 `SetFlag`，导致后续 `WaitFlag` 等到旧的 flag；
- `SetFlag` 在分支内、`WaitFlag` 在分支外；
- 用 `PipeBarrier<PIPE_ALL>` 替代 MTE/V/S 之间的显式同步；
- 跨核 flag 放在只有部分 block 执行的路径里。
