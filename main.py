"""
Interactive command-line entry point for the agent.

Runs a simple read-eval-print loop: reads a line from the user, hands it
(plus conversation history) to run_agent, prints the reply, repeats.
"""

from agent import SYSTEM_PROMPT, run_agent


def main():
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("Mini agent ready. Type 'exit' to quit.")
    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.strip().lower() in ("exit", "quit"):
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})
            reply = run_agent(messages)
            print(f"\nAgent: {reply}")
        except (KeyboardInterrupt, EOFError) as _:
            print("\nExiting agent session.")
            break


if __name__ == "__main__":
    main()
