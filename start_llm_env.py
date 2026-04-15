import subprocess
import requests

def flush_vram():
    qwen_models = [
        "http://localhost:11434/keep_alive",
        "http://localhost:11435/keep_alive"
    ]
    
    for model in qwen_models:
        try:
            response = requests.post(model, json={"keep_alive": 0})
            response.raise_for_status()
            print(f"VRAM flushed for {model}")
        except requests.RequestException as e:
            print(f"Failed to flush VRAM for {model}: {e}")

def start_docker_compose():
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("Docker Compose environment started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker Compose environment: {e}")

if __name__ == "__main__":
    flush_vram()
    start_docker_compose()
