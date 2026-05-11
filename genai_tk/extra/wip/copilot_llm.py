import json
import os
import subprocess
from typing import Any

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnableLambda

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class OpenCodeCopilotAdapter:
    """
    Corrected adapter using opencode auth login for GitHub Copilot
    """

    def __init__(self, model: str = "github-copilot/gpt-4o"):
        self.model = model
        self.auth_file = os.path.expanduser("~/.local/share/opencode/auth.json")
        self.authenticated = False

    def check_auth_status(self) -> bool:
        """
        Verify authentication status using correct command.
        """
        try:
            result = subprocess.run(["opencode", "auth", "ls"], capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Check if copilot is in the output
                if "copilot" in result.stdout.lower():
                    self.authenticated = True
                    return True
            return False
        except FileNotFoundError:
            print("⚠️  'opencode' CLI not found. Please install first.")
            return False
        except Exception as e:
            print(f"⚠️  Check failed: {e}")
            return False

    def authenticate(self) -> bool:
        """
        Run the correct authentication command.
        """
        try:
            print("🔐 Starting authentication flow...")
            print("Run this command in terminal:")
            print("  opencode auth login")
            print("  Then select 'GitHub Copilot' from the provider list")
            print()

            # Wait for user to complete auth
            input("Press Enter after authentication is complete...")

            return self.check_auth_status()
        except KeyboardInterrupt:
            return False

    def call_llm(self, system_prompt: str = "", user_prompt: str = "") -> dict[str, Any]:
        """
        Call LLM using OpenCode CLI.
        """
        if not self.check_auth_status():
            print("❌ Not authenticated. Please run: opencode auth login")
            return {"success": False, "error": "Authentication required"}

        try:
            message = f"{system_prompt}\n\n{user_prompt}".strip() if system_prompt else user_prompt
            cmd = ["opencode", "run", "--format", "json", "-m", self.model, message]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode != 0:
                return {"success": False, "error": result.stderr or result.stdout, "code": result.returncode}

            # Parse JSONL output – collect all 'text' event parts
            text_parts: list[str] = []
            for line in result.stdout.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    if event.get("type") == "text":
                        text_parts.append(event["part"]["text"])
                except json.JSONDecodeError:
                    pass

            content = "".join(text_parts)
            return {"success": True, "content": content}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_langchain_llm(self):
        """
        Return LangChain compatible Runnable.
        """
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain required")

        adapter = self

        def _invoke(messages) -> str:
            if isinstance(messages, list):
                prompt = "\n".join(m.content for m in messages)
            elif hasattr(messages, "messages"):
                prompt = "\n".join(m.content for m in messages.messages)
            elif hasattr(messages, "content"):
                prompt = messages.content
            else:
                prompt = str(messages)
            response = adapter.call_llm(user_prompt=prompt)
            return response.get("content", "")

        return RunnableLambda(_invoke)


def main():
    """
    Main execution flow with corrected commands.
    """
    adapter = OpenCodeCopilotAdapter()

    # Step 1: Check auth
    print("=" * 60)
    print("OpenCode GitHub Copilot Integration")
    print("=" * 60)

    if not adapter.check_auth_status():
        print("\n❌ Not authenticated with OpenCode")
        print("\nRun this command in your terminal:")
        print("  opencode auth login")
        print("\nThen select 'GitHub Copilot' as the provider")

        # Optional: auto-authenticate
        auth_confirm = input("\nTry to authenticate now? (y/n): ").lower()
        if auth_confirm == "y":
            if not adapter.authenticate():
                print("Authentication failed. Please check your credentials.")
                return
        else:
            return

    print("✅ Authentication verified")

    # Step 2: Test LLM
    print("\n🤖 Testing LLM...")
    response = adapter.call_llm(
        system_prompt="You are a helpful coding assistant.", user_prompt="Explain async/await in Python"
    )

    if response.get("success"):
        print(f"✅ Response: {response['content'][:500]}...")
    else:
        print(f"❌ Error: {response.get('error')}")

    # Step 3: LangChain Integration
    if HAS_LANGCHAIN:
        print("\n🦜🔗 LangChain Integration")
        llm = adapter.get_langchain_llm()
        template = ChatPromptTemplate.from_messages([("system", "You are a Python expert"), ("human", "{query}")])
        chain = template | llm | StrOutputParser()

        result = chain.invoke({"query": "Write a recursive Fibonacci function"})
        print(f"✅ Output:\n{result}")
    else:
        print("ℹ️ LangChain not installed")


if __name__ == "__main__":
    main()
