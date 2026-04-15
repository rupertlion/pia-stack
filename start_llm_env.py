import subprocess

def start_docker_compose():
    try:
        subprocess.run(["docker-compose", "up", "-d"], check=True)
        print("Docker Compose environment started successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Failed to start Docker Compose environment: {e}")

if __name__ == "__main__":
    start_docker_compose()
