import customtkinter as ctk
import tkinter as tk
from PIL import Image, ImageDraw, ImageTk
import math
import ctypes
from typing import Any

# Colors
COLOR = {
  "FG": "#e3e3e3",
  "TEXT": "#050505",
  "DOT": "#505050"
}

class LoadingAnimation(ctk.CTkFrame):
  """Custom loading animation."""
  def __init__(self, master, **kwargs: Any) -> None:
    """Initalize loading animation.
    Args:
      master (object): parent widget.
      **kwargs (Any): Additional keyword arguments passed to CTkFrame.
    """
    super().__init__(master, fg_color=COLOR["FG"], corner_radius=18, **kwargs)

    self.label = ctk.CTkLabel(
      self, 
      text="Thinking...", 
      text_color=COLOR["TEXT"],
      font=("Arial", 14)
    )
    self.label.pack(side="left", padx=(15, 0), pady=12)

    self.canvas_width = 50
    self.canvas_height = 30
    self.bg_color = COLOR["FG"]

    self.canvas = tk.Canvas(
      self, 
      width=self.canvas_width, 
      height=self.canvas_height, 
      bg=self.bg_color, 
      highlightthickness=0
    )
    self.canvas.pack(side="left", padx=(5, 15), pady=5)

    self.dot_radius = 3
    scale_factor = self._get_scaling_factor()
    self._image_ref = self.create_smooth_dot(self.dot_radius, COLOR["DOT"], scale_factor)

    self.dots = []
    self.spacing = 10
    self.center_y = self.canvas_height / 2
    self.amplitude = 3
    self.speed = 0.2
    self.time = 0
        
    start_x = 10
    for i in range(3):
      x = start_x + (i * self.spacing)
      dot_id = self.canvas.create_image(
        x, self.center_y, 
        image=self._image_ref, 
        anchor="center"
      )
      self.dots.append(dot_id)
            
    self.is_running = True
    self.animate()

  def _get_scaling_factor(self) -> float:
    """Retrieve scaling factor for current screen.
    Returns:
      float, scaling factor.
    """
    try:
      return ctypes.windll.shcore.GetScaleFactorForDevice(0) / 100
    except:
      return 1.0

  def create_smooth_dot(self, radius: int, color: str, scale: float) -> ImageTk.PhotoImage:
    """Create a circular dot image.
    Args:
      radius (int): radius of dot.
      color (str): color of the dot.
      scale (float): scaling factor.
    Returns:
      PhotoImage, dot image.
    """
    pixel_radius = int(radius * scale)
    size = pixel_radius * 2
    oversample = 4
    big_size = (size * oversample, size * oversample)
    
    img = Image.new("RGBA", big_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse((0, 0, big_size[0], big_size[1]), fill=color)
    
    img = img.resize((size, size), Image.Resampling.LANCZOS)
    return ImageTk.PhotoImage(img)

  def animate(self) -> None:
    """Play the animation of the dot."""
    if not self.is_running: return
    self.time += self.speed
    
    for i, dot_id in enumerate(self.dots):
      offset = math.sin(self.time + (i * 1.5)) * self.amplitude
      y = self.center_y + offset
      start_x = 10
      x = start_x + (i * self.spacing)
      self.canvas.coords(dot_id, x, y)
        
    self.after(20, self.animate)

  def stop(self) -> None:
    """Stop the animation and destroy the widget."""
    self.is_running = False
    self.destroy()