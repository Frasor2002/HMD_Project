from agent.agent import load_agent


def chat(model):
  """Interactive CLI session."""
  active = True

  while active:
    user_input = input(f"\nUser: ")
    if user_input.lower() == "quit()":
      print("Closing...")
      active = False
    else:
      response = model.chat(user_input)
      print(f"Assistant: {response}")


def main() -> None:
  """Execute the agent."""

  dialogue_agent = load_agent()
  chat(dialogue_agent)

if __name__ == "__main__":
  main()