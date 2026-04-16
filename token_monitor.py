import time
from typing import Dict

class TokenMonitor:
    def __init__(self):
        self.token_counts: Dict[str, int] = {}
        self.last_update_times: Dict[str, float] = {}

    def update_token_count(self, model_name: str, token_count: int):
        if model_name not in self.token_counts:
            self.token_counts[model_name] = 0
            self.last_update_times[model_name] = time.time()
        
        current_time = time.time()
        elapsed_time = current_time - self.last_update_times[model_name]
        
        if elapsed_time >= 1.0:  # Update every second
            tokens_per_second = (token_count - self.token_counts[model_name]) / elapsed_time
            print(f"Model {model_name}: {tokens_per_second:.2f} tokens/s")
            self.token_counts[model_name] = token_count
            self.last_update_times[model_name] = current_time

# Example usage:
if __name__ == "__main__":
    monitor = TokenMonitor()
    while True:
        # Simulate updating token counts for models
        monitor.update_token_count("model1", 100)
        time.sleep(0.5)
