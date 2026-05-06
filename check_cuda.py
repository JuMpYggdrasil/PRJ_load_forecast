import sys
import subprocess

def check_package(name):
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "installed")
    except ImportError:
        return None

def section(title):
    print(f"\n{'='*50}")
    print(f" {title}")
    print(f"{'='*50}")

def status(ok, msg):
    icon = "[OK]" if ok else "[FAIL]"
    print(f"  {icon} {msg}")
    return ok

def main():
    all_ok = True

    section("System Info")
    status(True, f"Python {sys.version}")
    status(True, f"Platform: {sys.platform}")

    section("GPU & Driver")
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.strip().split("\n"):
            status(True, line)
    except Exception as e:
        all_ok &= status(False, f"nvidia-smi failed: {e}")

    section("PyTorch & CUDA")
    import torch
    all_ok &= status(True, f"PyTorch {torch.__version__}")
    all_ok &= status(torch.cuda.is_available(), f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        all_ok &= status(True, f"CUDA version: {torch.version.cuda}")
        all_ok &= status(True, f"GPU count: {torch.cuda.device_count()}")
        all_ok &= status(True, f"GPU name: {torch.cuda.get_device_name(0)}")
        all_ok &= status(torch.cuda.device_count() >= 1, "At least 1 GPU detected")
    else:
        all_ok &= status(False, "No GPU detected")

    section("DRL Libraries")
    libs = {
        "gymnasium": "1.2.0",
        "stable_baselines3": "2.0.0",
        "numpy": "1.24.0",
        "pandas": "2.0.0",
        "matplotlib": "3.7.0",
    }
    for pkg, min_ver in libs.items():
        ver = check_package(pkg)
        if ver:
            status(True, f"{pkg} {ver}")
        else:
            all_ok &= status(False, f"{pkg} missing (min {min_ver})")

    section("SB3 Algorithms")
    try:
        from stable_baselines3 import PPO, A2C, DQN, SAC, TD3, DDPG
        status(True, "PPO, A2C, DQN, SAC, TD3, DDPG available")
    except ImportError as e:
        all_ok &= status(False, f"Missing algorithms: {e}")

    section("GPU Training Test")
    if torch.cuda.is_available():
        try:
            import gymnasium as gym
            from stable_baselines3 import PPO
            env = gym.make("CartPole-v1")
            model = PPO("MlpPolicy", env, device="cuda", verbose=0)
            model.learn(300)
            status(True, "PPO trained 300 steps on GPU successfully")
        except Exception as e:
            all_ok &= status(False, f"GPU training test failed: {e}")
    else:
        all_ok &= status(False, "Skipped - no GPU available")

    section("Summary")
    if all_ok:
        status(True, "Environment is ready for Deep RL!")
    else:
        status(False, "Some checks failed. Fix issues before starting Deep RL project.")

if __name__ == "__main__":
    main()
