from gui.chat import ChatGUI
from agent.agent import load_agent

if __name__ == "__main__":
  agent = load_agent()
  gui = ChatGUI(agent)
  gui.mainloop()
