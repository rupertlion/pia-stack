import subprocess
import httpx

def flush_vram():
    qwen_models = [
        {"model": "qwen2.5-coder:14b", "keep_alive": 0},
        {"model": "qwen3:32b", "keep_alive": 0}
    ]
    
    for model in qwen_models:
        try:
            response = httpx.post("http://localhost:11434/api/generate", json=model)
            response.raise_for_status()
            print(f"VRAM flushed for {model['model']}")
        except httpx.RequestException as e:
            print(f"Failed to flush VRAM for {model['model']}: {e}")

def start_docker_compose():
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("Docker Compose environment started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker Compose environment: {e}")

if __name__ == "__main__":
    flush_vram()
    start_docker_compose()
