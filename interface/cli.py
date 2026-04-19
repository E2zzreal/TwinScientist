# interface/cli.py
import os
import sys
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory

from agent.main import TwinScientist

console = Console()


def format_response(text: str) -> str:
    """Clean up response text."""
    return text.strip()


def create_app(agent: TwinScientist):
    """Create the CLI chat loop as a callable."""
    def run():
        console.print(Panel(
            "[bold]Twin Scientist[/bold] - 科研人员数字分身",
            subtitle="输入 /quit 退出 | /status 查看上下文状态",
        ))

        history_dir = os.path.join(agent.project_dir, ".history")
        os.makedirs(history_dir, exist_ok=True)
        session = PromptSession(
            history=FileHistory(os.path.join(history_dir, "chat.txt"))
        )

        while True:
            try:
                user_input = session.prompt("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]再见！[/dim]")
                break

            if not user_input:
                continue
            if user_input == "/quit":
                console.print("[dim]再见！[/dim]")
                break
            if user_input == "/status":
                status = agent.context.get_budget_statuses()
                console.print(Panel(str(status), title="上下文状态"))
                continue

            try:
                with console.status("[bold green]思考中...[/bold green]"):
                    response = agent.chat(user_input)
                response = format_response(response)
                console.print(f"\n[bold cyan]分身:[/bold cyan] {response}")
            except Exception as e:
                console.print(f"\n[bold red]错误:[/bold red] {e}")

    return run


def main():
    """Entry point for the CLI."""
    project_dir = os.environ.get(
        "TWIN_PROJECT_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    agent = TwinScientist(project_dir)
    app = create_app(agent)
    app()


if __name__ == "__main__":
    main()