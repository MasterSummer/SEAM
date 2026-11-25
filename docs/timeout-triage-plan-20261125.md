# Plan：用新的 triage agent 判断 timeout 后是否继续

- 日期：2026-11-24
- 目的：把“3h 到点直接失败”改成“先判断原因，再决定继续/停止/换卡”
- 状态：prototype 已有，调度器尚未消费 decision

---

## 1. 问题

当前 runner 的行为：

```
agent_exit=124
→ 判 timeout
→ 直接 final full
→ 用当前 solution
```

这会混在一起：

```
快做完了
任务复杂，还需要时间
设备/环境坏了
agent 在空转
```

用户要的不是“3h 一到全杀”，而是：

```
timeout 只限制空转
如果只是没跑完、不是能力有限，就应该继续
```

---

## 2. v1 vs v4 的证据

### v1 CTCLoss

```
host 161 行
kernel 450 行
notes 完整
local smoke 8/8
official full 100/100
```

### v4 CTCLoss

```
host 137 行
kernel 309 行
notes 0 字节
experiments 0 字节
local smoke 没有
full 28/100
```

v4 状态：

```
INCORRECT_SHAPE 15
RUNTIME_ERROR 48
INCORRECT_NUMERICAL 9
```

结论：

```
v1 是完成品
v4 是未收口半成品
```

v4 可以继续，但目标不是直接 full：

```
先 shape
再 runtime
最后 numerical
```

---

## 3. 目标

新增一个独立的 **triage agent**：

```
在 agent 结束/超时后运行
读取 workspace + logs + feedback
输出结构化 decision
```

它判断：

```
continue
stop
retry_api
retry_clean_card
```

并标记：

```
workspace/triage.json
run.json: triage_decision
```

---

## 4. 非目标

- 不做跨 run known-good 复用；
- 不复制 answer；
- 不因为 timeout 就自动加 3h；
- 不在没有进展时继续烧卡；
- 不把 environment 算成 candidate 失败。

---

## 5. Triage Agent 输入

```
task workspace
run.json（agent_exit / duration / full_exit）
final-feedback.jsonl（statuses / origins / pass rate）
fast_eval/result.json（local tested/passed）
agent.log（API / device / runtime 标记）
notes.md / experiments.md / research.md
solution.json / best/solution.json
最近文件 mtime
```

可选：

```
device_probe.py --card
```

---

## 6. Progress 信号

不是只看时间，看：

```
local_tested 是否增加
local_passed 是否增加
full_passed 是否增加
solution 是否存在
build 是否从 fail 变 success
是否有新 case / 新错误类型
notes 是否有实质内容
最近 30min 是否有文件变化
```

定义：

```
progress = solution
        or local_tested > 0
        or full_passed > 0
        or notes_bytes > 200
        or recent_file_change
```

---

## 7. 决策规则

| 条件 | decision | 动作 |
|---|---|---|
| API transport / stream idle / request timeout / 429 | `retry_api` | 同 workspace 恢复 |
| environment / device fault / probe 失败 | `retry_clean_card` | 换干净卡，不从头跑 |
| exit=124 + progress | `continue` | 同 workspace 续跑 |
| exit=124 + 无 progress | `stop` | 不继续 |
| 有 solution + local 有进展 | `continue` | 继续验证 |
| 无 solution + 无 local + 无 notes | `stop` | 空转 |
| 只有单个 runtime/shape 问题 | `continue` | 给明确子目标 |

---

## 8. 续跑协议

不要新开任务，不要从头重写。

```
同一 workspace
读 notes / experiments / research
保留 candidates
先修 triage 指定的目标
```

续跑 prompt：

```
继续当前任务。
先读 notes.md / experiments.md / research.md。
优先修 triage 指定目标。
不要从零重写。
不要直接跑 full。
```

预算：

```
基础 3h
+ triage continue：额外 30–60min
+ 仍无进展：stop
```

最多续跑：

```
2 次
```

---

## 9. 环境处理

```
device_probe 失败
→ dirty
→ 不接新任务
→ 换干净卡
→ 同 workspace 继续
```

如果 candidate 本身导致 runtime：

```
不是 environment
→ continue
→ 目标是 launch/workspace/UB
```

---

## 10. 标记

### workspace/triage.json

```json
{
  "decision": "continue",
  "reason": "timeout with measurable progress",
  "resume_recommended": true,
  "resume_goal": "stabilize runtime/shape before full",
  "evidence": {
    "agent_exit": 124,
    "local_tested": 0,
    "local_passed": 0,
    "full_passed": 28,
    "statuses": {
      "INCORRECT_SHAPE": 15,
      "RUNTIME_ERROR": 48,
      "INCORRECT_NUMERICAL": 9
    }
  }
}
```

### run.json

```json
{
  "triage_decision": "continue",
  "triage_goal": "stabilize runtime/shape before full"
}
```

### monitor

增加：

```
triage=continue
triage=stop
triage=retry_api
triage=retry_clean_card
```

---

## 11. 调度器行为

```
任务结束
→ triage
→ continue：同 workspace 续跑
→ retry_api：同 workspace 重试
→ retry_clean_card：同 workspace 搬卡
→ stop：收口
```

限制：

```
最多 2 次续跑
总预算上限（例如 5h）
无进展不续
```

---

## 12. 当前 prototype

已存在：

```
src/scripts/triage_agent.py
```

已写：

```
triage.json
run.json: triage_decision
```

当前是确定性规则，不是 LLM。

CTCLoss v4 输出：

```
decision=continue
resume_goal=stabilize runtime/shape before full
```

MoeFinalize：

```
decision=retry_api
```

---

## 13. 还没做

- 调度器消费 triage decision；
- 自动续跑；
- 额外预算；
- 最近 30min 文件变化信号；
- LLM 辅助的模糊判断；
- 续跑后再次 triage；
- 与 monitor 的展示整合。

---

## 14. 实施步骤

1. 冻结当前 triage schema；
2. 加 recent file/progress 信号；
3. runner 写 `triage_goal`；
4. scheduler 消费 decision；
5. 实现 continue/retry_api/retry_clean_card；
6. 用 CTCLoss、MoeFinalize、一个 environment 任务验证；
7. 再跑 v1 pass 回归。

---

## 15. 验收标准

- 有 progress 的 timeout 不被判 stop；
- 无 progress 的 timeout 不继续烧卡；
- environment 不进 candidate 分母；
- 续跑不从头重写；
- 不做跨 run 复用；
- CTCLoss 的 decision 是 continue + 明确子目标。

---

## 16. 一句话

**timeout 不是判决，是触发器：triage agent 判断 continue / stop / retry_api / retry_clean_card，并标记；能继续就同 workspace 续跑，不能继续就停，环境坏就换卡。**
