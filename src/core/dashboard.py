from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.ui_events import PHASE_DISPLAY

MAX_CURRENT_WORK = 140
MAX_ACTIVITY_LINE = 140
MAX_ACTIVITY_LINES = 6
STATUS_TEXT = {
    "pending": "待执行",
    "running": "运行中",
    "success": "已完成",
    "passed": "已完成",
    "skipped": "已跳过",
    "failed": "失败",
    "failure": "失败",
    "complete": "已结束",
    "dispatched": "已路由",
}


@dataclass
class DashboardState:
    phases: dict[str, dict[str, Any]] = field(default_factory=dict)
    current_work: str = "Waiting for workflow events..."
    activity: list[str] = field(default_factory=list)
    status: str = "running"


@dataclass(frozen=True)
class PhaseRow:
    number: int
    phase_id: str
    title: str
    status: str
    description: str


def _compact_text(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _short_phase_description(description: str) -> str:
    return _compact_text(description, 46)


def _status_text(status: object) -> str:
    return STATUS_TEXT.get(str(status or "pending"), str(status or "待执行"))


def visible_phase_rows(state: DashboardState) -> list[PhaseRow]:
    phase_ids = list(PHASE_DISPLAY)
    current_index = 0
    for index, phase_id in enumerate(phase_ids):
        status = str(state.phases.get(phase_id, {}).get("status", "pending"))
        if status == "running":
            current_index = index
            break
    else:
        for index, phase_id in enumerate(phase_ids):
            status = str(state.phases.get(phase_id, {}).get("status", "pending"))
            if status == "pending":
                current_index = index
                break
        else:
            current_index = max(len(phase_ids) - 1, 0)

    selected_ids = phase_ids[current_index : current_index + 2]
    rows: list[PhaseRow] = []
    for phase_id in selected_ids:
        copy = PHASE_DISPLAY[phase_id]
        status = state.phases.get(phase_id, {}).get("status", "pending")
        rows.append(
            PhaseRow(
                number=phase_ids.index(phase_id) + 1,
                phase_id=phase_id,
                title=copy.title,
                status=_status_text(status),
                description=_short_phase_description(copy.description),
            )
        )
    return rows


def _load_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int]:
    if not path.exists():
        return [], offset
    with path.open("r", encoding="utf-8") as handle:
        handle.seek(offset)
        lines = handle.readlines()
        new_offset = handle.tell()
    events: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(raw, dict):
            events.append(raw)
    return events, new_offset


def _apply_event(state: DashboardState, event: dict[str, Any]) -> None:
    event_type = str(event.get("event_type") or "")
    phase_id = event.get("phase_id")
    status = str(event.get("status") or "")
    message = _compact_text(event.get("message") or "", MAX_CURRENT_WORK)
    timestamp = str(event.get("timestamp") or "")[11:19]
    agent_role = event.get("agent_role")
    session_id = event.get("session_id")

    if isinstance(phase_id, str) and phase_id:
        phase = state.phases.setdefault(
            phase_id,
            {
                "title": PHASE_DISPLAY.get(phase_id, None).title
                if phase_id in PHASE_DISPLAY
                else phase_id,
                "description": PHASE_DISPLAY.get(phase_id, None).description
                if phase_id in PHASE_DISPLAY
                else "",
                "status": "pending",
            },
        )
        if event_type == "phase_started":
            phase["status"] = "running"
            state.current_work = f"{phase['title']}: {message or 'running'}"
        elif event_type == "phase_finished":
            phase["status"] = status or "finished"
            state.current_work = f"{phase['title']}: {status}"

    if event_type in {
        "agent_command_started",
        "agent_command_finished",
        "shell_command_started",
        "shell_command_finished",
        "session_ready",
        "opencode_tool_started",
        "opencode_phase_complete",
        "repair_iteration_started",
        "repair_iteration_finished",
    }:
        actor = str(agent_role or session_id or event_type)
        line = _compact_text(
            f"{timestamp} {actor} {status} {message}".strip(),
            MAX_ACTIVITY_LINE,
        )
        state.activity.append(line)
        state.activity = state.activity[-MAX_ACTIVITY_LINES:]
        if message:
            state.current_work = message

    if event_type == "workflow_finished":
        state.status = status or "complete"


def run_dashboard(events_path: str | Path, stop_event: threading.Event) -> None:
    """Render a best-effort real-time dashboard from ``ui_events.jsonl``."""
    try:
        from rich.console import Group
        from rich.live import Live
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except Exception:
        return

    path = Path(events_path)
    state = DashboardState()
    offset = 0

    def render() -> Group:
        table = Table(expand=True)
        table.add_column("编号", ratio=1)
        table.add_column("阶段", ratio=2)
        table.add_column("状态", ratio=1)
        table.add_column("正在做什么", ratio=3)
        for row in visible_phase_rows(state):
            table.add_row(str(row.number), row.title, row.status, row.description)
        activity = "\n".join(state.activity) or "暂无智能体活动。"
        return Group(
            Panel(Text(f"SEAM 迁移仪表盘  状态={_status_text(state.status)}"), title="运行"),
            Panel(table, title="当前阶段"),
            Panel(_compact_text(state.current_work, MAX_CURRENT_WORK), title="当前工作"),
            Panel(activity, title="智能体活动"),
            Panel("q: 退出仪表盘视图；迁移和日志继续运行", title="快捷键"),
        )

    with Live(render(), refresh_per_second=4, screen=True) as live:
        while not stop_event.is_set():
            events, offset = _load_events(path, offset)
            for event in events:
                _apply_event(state, event)
            live.update(render())
            time.sleep(0.25)


class SeamDashboardApp:
    """Small wrapper used by the runner to launch the live dashboard."""

    def __init__(self, events_path: str | Path, stop_event: threading.Event) -> None:
        self.events_path = Path(events_path)
        self.stop_event = stop_event

    def run(self) -> None:
        try:
            self._run_textual()
        except Exception:
            run_dashboard(self.events_path, self.stop_event)

    def _run_textual(self) -> None:
        from textual.app import App, ComposeResult
        from textual.containers import Container
        from textual.widgets import Footer, Header, Static

        events_path = self.events_path
        stop_event = self.stop_event

        class _TextualDashboard(App[None]):
            CSS = """
            Screen { layout: vertical; }
            #timeline { height: 45%; overflow-y: auto; }
            #current { height: 20%; }
            #activity { height: 1fr; overflow-y: auto; }
            """
            BINDINGS = [
                ("q", "quit", "Quit dashboard"),
                ("l", "focus_activity", "Logs"),
                ("s", "focus_activity", "Sessions"),
                ("?", "help", "Help"),
            ]

            def __init__(self) -> None:
                super().__init__()
                self.state = DashboardState()
                self.offset = 0

            def compose(self) -> ComposeResult:
                yield Header(show_clock=True)
                with Container():
                    yield Static("", id="timeline")
                    yield Static("", id="current")
                    yield Static("", id="activity")
                yield Footer()

            def on_mount(self) -> None:
                self.set_interval(0.25, self.refresh_events)

            def refresh_events(self) -> None:
                if stop_event.is_set():
                    self.exit()
                    return
                events, self.offset = _load_events(events_path, self.offset)
                for event in events:
                    _apply_event(self.state, event)
                self.query_one("#timeline", Static).update(self._timeline_text())
                self.query_one("#current", Static).update(
                    f"当前工作\n\n{_compact_text(self.state.current_work, MAX_CURRENT_WORK)}"
                )
                self.query_one("#activity", Static).update(
                    "智能体活动\n\n"
                    + ("\n".join(self.state.activity) or "暂无智能体活动。")
                )

            def _timeline_text(self) -> str:
                lines = ["当前阶段"]
                for row in visible_phase_rows(self.state):
                    lines.append(
                        f"{row.number}. {row.title}｜{row.status}\n"
                        f"   {row.description}"
                    )
                return "\n".join(lines)

            def action_focus_activity(self) -> None:
                self.query_one("#activity", Static).focus()

            def action_help(self) -> None:
                self.query_one("#current", Static).update(
                    "快捷键\n\nq: 退出仪表盘视图\n"
                    "l/s: 聚焦智能体活动面板\n?: 显示帮助\n"
                    "退出仪表盘后，迁移任务仍会继续运行。"
                )

        _TextualDashboard().run()
