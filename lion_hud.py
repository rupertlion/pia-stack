import tkinter as tk
import requests
import subprocess
import threading
import sys
import time

# Configuration
OLLAMA_HOST = "http://localhost:11434"
PRIMARY_MODEL = "PIA-Coder"

class LionHUD:
    def __init__(self, root):
        self.root = root
        self.root.title("LION 5090 COMMAND CENTER")
        self.root.geometry("480x420")
        self.root.configure(bg='#121212')
        self.root.attributes("-topmost", True)

        self.is_running = True
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # --- ZONE 1: TELEMETRY ---
        status_frame = tk.Frame(self.root, bg='#1a1a1a', bd=1, relief="solid")
        status_frame.pack(pady=10, padx=10, fill="x")

        self.model_label = tk.Label(status_frame, text="Model: Scanning...", fg="#00ff00", bg="#1a1a1a", font=("Consolas", 12, "bold"))
        self.model_label.pack(pady=5)

        self.vram_label = tk.Label(status_frame, text="VRAM: -- / 32 GB", fg="#00d4ff", bg="#1a1a1a", font=("Consolas", 11))
        self.vram_label.pack()

        self.speed_label = tk.Label(status_frame, text="SPEED: 0.00 t/s", fg="#ffaa00", bg="#1a1a1a", font=("Consolas", 24, "bold"))
        self.speed_label.pack(pady=10)

        # --- ZONE 2: CONTROLS ---
        btn_frame = tk.Frame(self.root, bg='#121212')
        btn_frame.pack(fill="both", expand=True, padx=20)

        # Load Primary
        tk.Button(btn_frame, text="LOAD PIA-CODER", command=lambda: self.pre_warm(PRIMARY_MODEL), 
                  bg="#222222", fg="white", font=("Consolas", 10)).pack(fill="x", pady=2)

        # Gaming Flush
        tk.Button(btn_frame, text="FLUSH VRAM (GAMING MODE)", command=self.flush, 
                  bg="#4a1a1a", fg="#ff5555", font=("Consolas", 10, "bold")).pack(fill="x", pady=10)

        # Systemctl controls
        sys_frame = tk.Frame(btn_frame, bg='#121212')
        sys_frame.pack(fill="x")
        tk.Button(sys_frame, text="START ENGINE", command=lambda: self.sys_cmd("start"), bg="#1a3a1a", fg="#55ff55").pack(side="left", expand=True, fill="x")
        tk.Button(sys_frame, text="STOP ENGINE", command=lambda: self.sys_cmd("stop"), bg="#3a1a1a", fg="#ff5555").pack(side="right", expand=True, fill="x")

        self.last_ts = 0
        self.current_model = None
        self.update_loop()

    def on_closing(self):
        self.is_running = False
        self.root.destroy()
        sys.exit(0)

    def update_loop(self):
        if not self.is_running: return
        
        # Poll API
        try:
            res = requests.get(f"{OLLAMA_HOST}/api/ps", timeout=0.3).json()
            if res.get('models'):
                self.current_model = res['models'][0]['name']
                self.model_label.config(text=f"ACTIVE: {self.current_model}")
            else:
                self.current_model = None
                self.model_label.config(text="IDLE", fg="#888888")
        except: self.model_label.config(text="OFFLINE", fg="red")

        # Poll Hardware
        try:
            smi = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True).strip()
            vram, load = smi.split(',')
            self.vram_label.config(text=f"VRAM: {round(int(vram)/1024, 1)} / 32.0 GB")
            # Simple heuristic for speed counter
            if int(load) > 10: self.speed_label.config(text="GENERATING...", fg="#00ff00")
            else: self.speed_label.config(text="STANDBY", fg="#ffaa00")
        except: pass

        if self.is_running: self.root.after(800, self.update_loop)

    def pre_warm(self, m): threading.Thread(target=lambda: requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": m, "keep_alive": "5m"}), daemon=True).start()
    def flush(self): 
        if self.current_model: threading.Thread(target=lambda: requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": self.current_model, "keep_alive": 0}), daemon=True).start()
    def sys_cmd(self, op): threading.Thread(target=lambda: subprocess.run(["sudo", "systemctl", op, "ollama"]), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = LionHUD(root)
    root.mainloop()