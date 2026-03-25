"""Terminal UI — split pane with user chat (left) and inner thoughts (right)."""

from __future__ import annotations

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Header, Footer, RichLog, Input, Static
from textual.binding import Binding


class ThoughtsPanel(RichLog):
    """Right pane — shows the internal message board in real time."""
    pass


class ChatLog(RichLog):
    """Left pane — shows the user-facing conversation."""
    pass


class PsycheApp(App):
    """Main application."""

    CSS = """
    #main {
        height: 1fr;
    }
    #chat-pane {
        width: 1fr;
        border: solid $primary;
        height: 100%;
    }
    #thoughts-pane {
        width: 1fr;
        border: solid $secondary;
        height: 100%;
    }
    #chat-container {
        width: 1fr;
        height: 100%;
    }
    #thoughts-container {
        width: 1fr;
        height: 100%;
    }
    ChatLog {
        height: 1fr;
    }
    ThoughtsPanel {
        height: 1fr;
    }
    #chat-label, #thoughts-label {
        height: 1;
        background: $surface;
        color: $text;
        text-align: center;
        text-style: bold;
    }
    #user-input {
        dock: bottom;
        height: 3;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, on_user_input: callable | None = None):
        super().__init__()
        self._on_user_input = on_user_input

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main"):
            with Vertical(id="chat-container"):
                yield Static("Chat", id="chat-label")
                yield ChatLog(id="chat-pane", wrap=True, highlight=True, markup=True)
                yield Input(placeholder="Type a message...", id="user-input")
            with Vertical(id="thoughts-container"):
                yield Static("Inner Thoughts", id="thoughts-label")
                yield ThoughtsPanel(
                    id="thoughts-pane", wrap=True, highlight=True, markup=True
                )
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        if not text:
            return
        event.input.value = ""
        chat = self.query_one("#chat-pane", ChatLog)
        chat.write(f"[bold cyan]You:[/] {text}")
        if self._on_user_input:
            self._on_user_input(text)

    def post_chat(self, author: str, message: str) -> None:
        """Add a message to the chat pane (called from async code)."""
        try:
            chat = self.query_one("#chat-pane", ChatLog)
            chat.write(f"[bold green]{author}:[/] {message}")
        except Exception:
            pass

    def post_thought(self, author: str, message: str, urgency: float, importance: float) -> None:
        """Add a thought to the inner thoughts pane."""
        try:
            panel = self.query_one("#thoughts-pane", ThoughtsPanel)
            color = self._author_color(author)
            label = self._author_label(author)
            panel.write(f"[bold {color}]{label}:[/] {message}")
        except Exception:
            pass

    @staticmethod
    def _author_color(author: str) -> str:
        colors = {
            "perception": "bright_yellow",
            "emotion": "bright_red",
            "reasoning": "bright_blue",
            "memory": "bright_magenta",
            "self-model": "bright_cyan",
            "social": "bright_green",
            "drive": "orange1",
            "critic": "grey70",
            "orchestrator": "bold white",
        }
        return colors.get(author, "white")

    @staticmethod
    def _author_label(author: str) -> str:
        labels = {
            "perception": "PERCEPTION",
            "emotion": "EMOTION",
            "reasoning": "REASONING",
            "memory": "MEMORY",
            "self-model": "SELF-MODEL",
            "social": "SOCIAL",
            "drive": "DRIVE",
            "critic": "CRITIC",
            "orchestrator": "ORCHESTRATOR",
        }
        return labels.get(author, author.upper())
