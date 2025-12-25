import customtkinter as ctk
import threading
from PIL import Image, ImageDraw, ImageOps
import os
from gui.loading import LoadingAnimation
from agent.agent import DialogueAgent

GUI_DIR = os.path.dirname(os.path.abspath(__file__))
BOT_ICON_PATH = os.path.join(GUI_DIR, "assets", "bot_icon.png")
ICON_PATH =os.path.join(GUI_DIR, "assets", "icon.ico")

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
START_MSG = "Hi, I am a virtual assistant with the goal of helping out with videogame-related needs.\nYou can start by asking me something.\nBy writing \"HELP\" I can also give you an explanation of my capabilities.\nLet's get started!"
HELP_MSG= "I am able to answer several requests:\n1. searching for information on a game\n2. searching for a fun new game to play\n3. comparing two games to help you choose\n4.Handle your wishlist\n5. Explain gaming terminology\n6. Telling your friend games"


class ChatGUI(ctk.CTk):
  """GUI for the chat with the dialogue agent."""
  def __init__(self, agent: DialogueAgent):
    super().__init__()
    self.agent = agent

    self.title("HMD Project")
    ctk.set_appearance_mode("Light") 
    self.configure(fg_color=COLOR["FG"])
    self.iconbitmap(ICON_PATH)

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
      text="Human Machine Dialogue Project", 
      font=("Arial", 20, "bold"),
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
      text_color=COLOR["PLACEHOLDER"],
      corner_radius=20,
      font=("Arial", 14),
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


  def on_global_click(self, event):
    if event.widget != self.input_box._textbox:
      self.focus()
        

  def load_circular_icon(self, image_path):
    display_size = (52, 52)
    process_size = (150, 150) 

    img = Image.open(image_path).convert("RGBA")
        
    img = ImageOps.fit(img, process_size, method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    mask = Image.new('L', process_size, 0)
    draw = ImageDraw.Draw(mask) 
    draw.ellipse((0, 0) + process_size, fill=255)
    img.putalpha(mask)
        
    return ctk.CTkImage(light_image=img, dark_image=img, size=display_size)

    
  def on_entry_click(self, event):
    current_text = self.input_box.get("1.0", "end-1c")
    if current_text == self.placeholder_text:
      self.input_box.delete("1.0", "end")
      self.input_box.configure(text_color=COLOR["TEXT"])

  def on_focus_out(self, event):
    current_text = self.input_box.get("1.0", "end-1c")
    if current_text.strip() == "":
      self.input_box.delete("1.0", "end")
      self.input_box.insert("0.0", self.placeholder_text)
      self.input_box.configure(text_color=COLOR["PLACEHOLDER"])

    
  def reset_chat(self):
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


  def on_enter_pressed(self, event):
    self.send_message()
    return "break" 

  def set_input_state(self, state):
    if state == "disabled":
      self.input_box.configure(state="disabled", fg_color=COLOR["DISABLED"])
    else:
      self.input_box.configure(state="normal", fg_color=COLOR["MSG_BG"])


  def send_message(self, event=None):
    # Get response
    text = self.input_box.get("1.0", "end-1c")
        
    if text.strip() != "" and text != self.placeholder_text:
      self.add_message(text.strip(), is_bot=False)
            
      self.input_box.delete("1.0", "end")
            
      self.focus()
            
      self.set_input_state("disabled")
            
      self.show_loading()
      threading.Thread(target=self.process_backend_logic, args=(text,), daemon=True).start()


  def process_backend_logic(self, text):
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

  def display_bot_response(self, response):
    self.hide_loading()
    self.add_message(response, is_bot=True)
    self.set_input_state("normal")
    self.input_box.insert("0.0", self.placeholder_text)
    self.input_box.configure(text_color=COLOR["PLACEHOLDER"])

  def _on_mouse_wheel(self, event):
    current_pos = self.chat_history._parent_canvas.yview()
    if event.delta > 0 and current_pos[0] <= 0.0: return 
    self.chat_history._parent_canvas.yview_scroll(int(-20*(event.delta/120)), "units")

  def _on_arrow_key(self, direction):
    current_pos = self.chat_history._parent_canvas.yview()
    if direction < 0 and current_pos[0] <= 0.0: return
    if direction > 0 and current_pos[1] >= 1.0: return
    self.chat_history._parent_canvas.yview_scroll(direction * 30, "units")

  def scroll_to_bottom(self):
    self.chat_history.update_idletasks() 
    self.after(10, lambda: self.chat_history._parent_canvas.yview_moveto(1.0))

  def show_loading(self):
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

  def hide_loading(self):
    if self.loading_animation is not None:
      self.loading_animation.stop()
      if self.loading_container:
        self.loading_container.destroy()
      self.loading_animation = None
      self.loading_container = None


  def add_message(self, text, is_bot):
    msg_container = ctk.CTkFrame(self.chat_history, fg_color=COLOR["FG"])
    msg_container.pack(fill="x", pady=5, padx=10)
    wrap_len = 550 
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
        corner_radius=18, 
        wraplength=wrap_len, 
        justify="left",
        font=("Arial", 14)
      )
      bubble.grid(row=0, column=1, sticky="w", ipadx=12, ipady=12)
    else: # User msg
      bubble = ctk.CTkLabel(
        msg_container, 
        text=text, 
        fg_color=COLOR["MSG_BG"], 
        bg_color=COLOR["FG"],
        text_color=COLOR["TEXT"], 
        corner_radius=18, 
        wraplength=wrap_len, 
        justify="left",
        font=("Arial", 14)
      )
      bubble.pack(side="right", anchor="e", ipadx=12, ipady=12)
        
    self.scroll_to_bottom()