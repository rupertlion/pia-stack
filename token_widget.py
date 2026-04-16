import tkinter as tk
from typing import Dict
import time

class TokenWidget:
    def __init__(self, master: tk.Tk):
        self.master = master
        self.master.title("Token Speed-o-Meter")
        
        self.token_labels: Dict[str, tk.Label] = {}
        self.token_counts: Dict[str, int] = {}
        self.last_update_times: Dict[str, float] = {}

    def add_model(self, model_name: str):
        label = tk.Label(self.master, text=f"{model_name}: 0.00 tokens/s")
        label.pack()
        self.token_labels[model_name] = label
        self.token_counts[model_name] = 0
        self.last_update_times[model_name] = time.time()

    def update_token_count(self, model_name: str, token_count: int):
        current_time = time.time()
        elapsed_time = current_time - self.last_update_times[model_name]
        
        if elapsed_time >= 1.0:  # Update every second
            tokens_per_second = (token_count - self.token_counts[model_name]) / elapsed_time
            self.token_labels[model_name].config(text=f"{model_name}: {tokens_per_second:.2f} tokens/s")
            self.token_counts[model_name] = token_count
            self.last_update_times[model_name] = current_time

# Example usage:
if __name__ == "__main__":
    root = tk.Tk()
    widget = TokenWidget(root)
    widget.add_model("model1")

    def update_tokens():
        # Simulate updating token counts for models
        widget.update_token_count("model1", 100)
        root.after(500, update_tokens)

    update_tokens()
    root.mainloop()
