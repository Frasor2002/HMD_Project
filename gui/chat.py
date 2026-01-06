import customtkinter as ctk
import threading
from PIL import Image, ImageDraw, ImageOps, ImageTk
import os
from gui.loading import LoadingAnimation
from agent.agent import DialogueAgent
from typing import Any

ctk.set_widget_scaling(1.0)
ctk.set_window_scaling(1.0)

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ICON_PATH = os.path.join(GUI_DIR, "assets", "bot_icon.png")
ICON_PATH =os.path.join(GUI_DIR, "assets", "icon.ico")

FONT = "Liberation Sans"

COLOR = {
  "FG": "#e3e3e3",
  "HEADER": "#cfcfcf",
  "TEXT": "#050505",
  "RESET": "#353434",
  "RESET_HOV": "#7e7d7d",
  "PLACEHOLDER": "#65676b",
  "MSG_BG": "#ffffff",
  "DISABLED": "#ffffff",
  "BOT_MSG": "#d1d1d1"
}

PLACEHOLDER_TXT = "Write you request..."
START_MSG = "Hi! I'm your virtual gaming assistant. Feel free to ask me anything about video games.\n\nNot sure where to start? Just type \"help\" to see what I can do.\n\nLet's get started!"
HELP_MSG = "Here is what I can do for you:\n1. Search for information on a game\n2. Find a fun new game to play\n3. Compare two games to help you choose\n4. Manage your wishlist\n5. Explain gaming terminology\n6. See your friends' games"
INPUT_CORNER_RAD = 20
MSG_CORNER_RAD = 18

class ChatGUI(ctk.CTk):
  """GUI for the chat with the dialogue agent."""
  def __init__(self, agent: DialogueAgent) -> None:
    """Initialize the gui.
    Args:
      agent (DialogueAgent): dialogue agent to run.
    """
    super().__init__()
    self.agent = agent

    self.title("HMD Project")
    ctk.set_appearance_mode("Light") 
    self.configure(fg_color=COLOR["FG"])
    try:
      # On linux wont work
      self.iconbitmap(ICON_PATH)
    except Exception as e:
      print(f"Icon not loaded: {e}.")

    # Set window dimention and position
    window_width = 800
    window_height = 600
    screen_width = self.winfo_screenwidth()
    screen_height = self.winfo_screenheight()
    center_x = (screen_width // 2) - (window_width // 2)
    center_y = (screen_height // 2) - (window_height // 2)
    self.geometry(f"{window_width}x{window_height}+{center_x}+{center_y - 50}")

    # Layout
    self.grid_rowconfigure(1, weight=1) 
    self.grid_columnconfigure(0, weight=1)

    # Bot icon
    self.bot_icon = self.load_circular_icon(BOT_ICON_PATH)

    # Header
    self.header = ctk.CTkFrame(self, fg_color=COLOR["HEADER"], height=60, corner_radius=0)
    self.header.grid(row=0, column=0, sticky="nsew")
    self.header.grid_columnconfigure(1, weight=1) 
    self.header_label = ctk.CTkLabel(
      self.header, 
      text="Human-Machine Dialogue Project", 
      font=(FONT, 20, "bold"),
      text_color=COLOR["TEXT"]
    )
    self.header_label.grid(row=0, column=0, padx=20, pady=15, sticky="w")
    
    # Button to reset the chat
    self.reset_btn = ctk.CTkButton(
      self.header,
      text="Reset",
      fg_color=COLOR["RESET"],
      hover_color=COLOR["RESET_HOV"],
      width=80,
      command=self.reset_chat
    )
    self.reset_btn.grid(row=0, column=2, padx=20, sticky="e")
    
    # Chat history
    self.chat_history = ctk.CTkScrollableFrame(
      self, 
      fg_color=COLOR["FG"],
      scrollbar_button_color=COLOR["FG"], 
      scrollbar_button_hover_color=COLOR["FG"],
      corner_radius=0
    )
    self.chat_history.grid(row=1, column=0, sticky="nsew", padx=0, pady=0)
    self.chat_history._scrollbar.configure(width=0)

    # Text input
    self.input_container = ctk.CTkFrame(self, fg_color=COLOR["FG"])
    self.input_container.grid(row=2, column=0, sticky="ew", padx=20, pady=(0,20))
    self.input_container.grid_columnconfigure(0, weight=1) 
    self.input_container.grid_columnconfigure(2, weight=1) 
    self.input_box = ctk.CTkTextbox(
      self.input_container, 
      height=100, 
      width=700, 
      border_width=0, 
      border_color=COLOR["MSG_BG"],
      fg_color=COLOR["MSG_BG"],
      bg_color=COLOR["FG"],
      text_color=COLOR["PLACEHOLDER"],
      corner_radius=INPUT_CORNER_RAD,
      font=(FONT, 14),
      wrap="word"
    )
    self.input_box.grid(row=0, column=1, sticky="ew")
    # Placeholder text
    self.placeholder_text = PLACEHOLDER_TXT
    self.input_box.insert("0.0", self.placeholder_text)
        
    # Bindings
    self.input_box.bind("<Return>", self.on_enter_pressed)
    self.input_box.bind("<FocusIn>", self.on_entry_click)
    self.input_box.bind("<FocusOut>", self.on_focus_out)
    self.bind_all("<Button-1>", self.on_global_click)
        
    self.chat_history._parent_canvas.bind_all("<MouseWheel>", self._on_mouse_wheel)
    self.bind("<Up>", lambda e: self._on_arrow_key(-1))
    self.bind("<Down>", lambda e: self._on_arrow_key(1))

    self.loading_animation = None
    self.loading_container = None

    # Add starting message of the bot
    self.add_message(START_MSG, is_bot=True)


  def on_global_click(self, event: Any) -> None:
    """Handle global mouse clicks to handle focus.
    Args:
      event (Any): event object with widget info.
    """
    if event.widget != self.input_box._textbox:
      self.focus()
        

  def load_circular_icon(self, image_path: str) -> ctk.CTkImage:
    """Load and process an image into a circular icon.
    Args:
      image_path (str): path of the image to load.
    Returns:
      CTkImage, circular icon.
    """
    display_size = (52, 52)
    process_size = (150, 150) 

    img = Image.open(image_path).convert("RGBA")
        
    img = ImageOps.fit(img, process_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    mask = Image.new('L', process_size, 0)
    draw = ImageDraw.Draw(mask) 
    draw.ellipse((0, 0) + process_size, fill=255)
    img.putalpha(mask)
        
    return ctk.CTkImage(light_image=img, dark_image=img, size=display_size)

    
  def on_entry_click(self, event: Any) -> None:
    """Handle click on the input form the remove placeholder text.
    Args:
      event (Any): focus event.
    """
    current_text = self.input_box.get("1.0", "end-1c")
    if current_text == self.placeholder_text:
      self.input_box.delete("1.0", "end")
      self.input_box.configure(text_color=COLOR["TEXT"])

  def on_focus_out(self, event: Any) -> None:
    """Handle focus event on input box to restore placeholder.
    Args:
      event (Any): focus event.
    """
    current_text = self.input_box.get("1.0", "end-1c")
    if current_text.strip() == "":
      self.input_box.delete("1.0", "end")
      self.input_box.insert("0.0", self.placeholder_text)
      self.input_box.configure(text_color=COLOR["PLACEHOLDER"])

    
  def reset_chat(self) -> None:
    """Reset chat history and agent state."""
    self.agent.clear_history()

    self.input_box.delete("1.0", "end")
    self.input_box.insert("0.0", self.placeholder_text)
    self.input_box.configure(text_color=COLOR["PLACEHOLDER"])
    self.set_input_state("normal")
        
    self.hide_loading()
        
    for widget in self.chat_history.winfo_children():
      widget.destroy()
            
    self.chat_history.update_idletasks()
    self.chat_history._parent_canvas.yview_moveto(0.0)    
    self.add_message(START_MSG, is_bot=True)


  def on_enter_pressed(self, event: Any) -> str:
    """Handle enter key pressed to send message.
    Args:
      event (Any): the keypress event.
    Returns:
      str, "break" string to stop event propagation.
    """
    self.send_message()
    return "break" 

  def set_input_state(self, state: str) -> None:
    """Set state of the input box.
    Args:
      state (str): state of the input box.
    """
    if state == "disabled":
      self.input_box.configure(state="disabled", fg_color=COLOR["DISABLED"])
      self.reset_btn.configure(state="disabled")
    else:
      self.input_box.configure(state="normal", fg_color=COLOR["MSG_BG"])
      self.reset_btn.configure(state="normal")


  def send_message(self, event: Any =None) -> None:
    """Extract text from input and initiate msg processing.
    Args:
      event (Any): event triggering the send.
    """
    # Get response
    text = self.input_box.get("1.0", "end-1c")
        
    if text.strip() != "" and text != self.placeholder_text:
      self.add_message(text.strip(), is_bot=False)
            
      self.input_box.delete("1.0", "end")
            
      self.focus()
            
      self.set_input_state("disabled")
            
      self.show_loading()
      threading.Thread(target=self.process_backend_logic, args=(text,), daemon=True).start()


  def process_backend_logic(self, text: str) -> None:
    """Process user input ina separate thread.
    Args:
      text (str), user input txt.
    """
    # Catch help message
    if text.lower() == "help":
      response = HELP_MSG
    else:
      try:
        response = self.agent.chat(text)
        print(response)
      except Exception as e:
        response = f"Error processing request: {str(e)}"
    # Update UI after we are done
    self.after(0, self.display_bot_response, response)

  def display_bot_response(self, response: str) -> None:
    """Update UI with the bot's response.
    Args:
      response (str), response from the bot.
    """
    self.hide_loading()
    self.add_message(response, is_bot=True)
    self.set_input_state("normal")
    self.input_box.insert("0.0", self.placeholder_text)
    self.input_box.configure(text_color=COLOR["PLACEHOLDER"])

  def _on_mouse_wheel(self, event: Any) -> None:
    """Handle mouse weel scrolling events.
    Args:
      event (Any), mouse weel event.
    """
    current_pos = self.chat_history._parent_canvas.yview()
    if event.delta > 0 and current_pos[0] <= 0.0: return 
    self.chat_history._parent_canvas.yview_scroll(int(-20*(event.delta/120)), "units")

  def _on_arrow_key(self, direction: int) -> None:
    """Handle arrow key scrolling event.
    Args:
      direction (int): direction where scrolling happens.
    """
    current_pos = self.chat_history._parent_canvas.yview()
    if direction < 0 and current_pos[0] <= 0.0: return
    if direction > 0 and current_pos[1] >= 1.0: return
    self.chat_history._parent_canvas.yview_scroll(direction * 30, "units")

  def scroll_to_bottom(self) -> None:
    """Scroll to bottom of chat history."""
    self.chat_history.update_idletasks() 
    self.after(10, lambda: self.chat_history._parent_canvas.yview_moveto(1.0))

  def show_loading(self) -> None:
    """Display loading animation bubble."""
    if self.loading_animation: return 
        
    self.loading_container = ctk.CTkFrame(self.chat_history, fg_color=COLOR["FG"])
    self.loading_container.pack(fill="x", pady=5, padx=10)
    self.loading_container.grid_columnconfigure(1, weight=1)
    icon_label = ctk.CTkLabel(self.loading_container, text="", image=self.bot_icon)
    icon_label.grid(row=0, column=0, sticky="e", padx=(0, 10))

    self.loading_animation = LoadingAnimation(self.loading_container)
    self.loading_animation.grid(row=0, column=1, sticky="w")
    # This will scroll down when thinking starts
    self.scroll_to_bottom()

  def hide_loading(self) -> None:
    """Remove loading animation bubble."""
    if self.loading_animation is not None:
      self.loading_animation.stop()
      if self.loading_container:
        self.loading_container.destroy()
      self.loading_animation = None
      self.loading_container = None


  def add_message(self, text: str, is_bot: bool) -> None:
    """Add new message to chat history.
    Args:
      text (str): message text.
      is_bot (bool): flag to choose who is the sender.
    """

    msg_container = ctk.CTkFrame(self.chat_history, fg_color=COLOR["FG"])
    msg_container.pack(fill="x", pady=5, padx=10)
    wrap_len = 500 
    if is_bot:
      msg_container.grid_columnconfigure(1, weight=1)
      icon_label = ctk.CTkLabel(msg_container, text="", image=self.bot_icon)
      icon_label.grid(row=0, column=0, sticky="e", padx=(0, 10))
      bubble = ctk.CTkLabel(
        msg_container, 
        text=text, 
        fg_color=COLOR["BOT_MSG"], 
        bg_color=COLOR["FG"],  
        text_color=COLOR["TEXT"], 
        corner_radius=MSG_CORNER_RAD,
        wraplength=wrap_len, 
        justify="left",
        font=(FONT, 14)
      )
      bubble.grid(row=0, column=1, sticky="w", ipadx=12, ipady=12)
    else: # User msg
      bubble = ctk.CTkLabel(
        msg_container, 
        text=text, 
        fg_color=COLOR["MSG_BG"], 
        bg_color=COLOR["FG"],
        text_color=COLOR["TEXT"], 
        corner_radius=MSG_CORNER_RAD,
        wraplength=wrap_len, 
        justify="left",
        font=(FONT, 14)
      )
      bubble.pack(side="right", anchor="e", ipadx=12, ipady=12)
        
    self.scroll_to_bottom()