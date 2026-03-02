import os
import subprocess
from typing import Any, Dict

try:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


class OpenCodeCopilotAdapter:
    """
    Corrected adapter using opencode auth login for GitHub Copilot
    """

    def __init__(self, model: str = "copilot"):
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

    def call_llm(self, system_prompt: str = "", user_prompt: str = "") -> Dict[str, Any]:
        """
        Call LLM using OpenCode CLI.
        """
        if not self.check_auth_status():
            print("❌ Not authenticated. Please run: opencode auth login")
            return {"success": False, "error": "Authentication required"}

        try:
            # Run OpenCode with prompt
            cmd = ["opencode", "-p", f"{system_prompt}\n\n{user_prompt}"]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

            if result.returncode == 0:
                return {"success": True, "content": result.stdout}
            else:
                return {"success": False, "error": result.stderr, "code": result.returncode}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_langchain_llm(self):
        """
        Return LangChain compatible object.
        """
        if not HAS_LANGCHAIN:
            raise ImportError("LangChain required")

        class CopilotLLM:
            def __init__(self, adapter):
                self.adapter = adapter

            def invoke(self, prompt: str) -> str:
                response = self.adapter.call_llm(user_prompt=prompt)
                return response.get("content", "")

        return CopilotLLM(self)


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
