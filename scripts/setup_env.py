# ==============================================================================
# 🚀 REPOSITORY SETUP SCRIPT (RUN ONCE)
# Safely manages Git cloning for external dependencies (e.g., Kwai-Kolors MPS).
# ==============================================================================

import subprocess
import sys
import os

def setup_master_environment():
    print("⏳ Initializing external repositories. Please wait...")
    
    # ---------------------------------------------------------
    # GIT CLONE HANDLING (For Kwai-Kolors MPS)
    # ---------------------------------------------------------
    mps_repo_url = "https://github.com/Kwai-Kolors/MPS.git"
    
    # Dynamic path routing: resolves to the project root directory
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mps_local_path = os.path.join(base_dir, "MPS")
    
    # Idempotent cloning: Only clone if the directory doesn't exist
    if not os.path.exists(mps_local_path):
        print(f"📦 Cloning the MPS repository to {mps_local_path}...")
        subprocess.run(["git", "clone", mps_repo_url, mps_local_path], check=True)
    else:
        print("✅ MPS directory already exists. Skipping clone step.")
        
    # Append MPS to the system path to enable cross-module importing
    if mps_local_path not in sys.path:
        sys.path.append(mps_local_path)
        
    print("🚀 SETUP COMPLETE! All external repositories are ready.")

if __name__ == "__main__":
    setup_master_environment()