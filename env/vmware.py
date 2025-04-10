import os
import subprocess
import time
from dotenv import load_dotenv


class VmwareController:
    def __init__(self):
        load_dotenv()
        self.remote_host = os.getenv("REMOTE_HOST")
        self.remote_user = os.getenv("REMOTE_USER")
        self.vmx_path = os.getenv("VMX_PATH")
        self.snapshot_name = os.getenv("SNAPSHOT_NAME")
        self.bg = None

    def run_remote_bg(self, cmd):
        ssh_cmd = ["ssh", "-tt", f"{self.remote_user}@{self.remote_host}"] + cmd
        proc = subprocess.Popen(ssh_cmd)
        return proc

    def run_remote(self, cmd):
        ssh_cmd = ["ssh", f"{self.remote_user}@{self.remote_host}"] + cmd
        return subprocess.run(ssh_cmd, check=True)

    def reset(self):
        start = time.time()

        # 1. Stop the VM if it's running
        try:
            print("Stopping VM...")
            self.run_remote(["vmrun", "stop", self.vmx_path, "hard"])
            time.sleep(5)
        except subprocess.CalledProcessError as e:
            print("Error during stopping the VM:", e)

        try:
            # Kill the previous ssh session if open
            if self.bg is not None:
                print("Killing the previous background process...")
                self.bg.terminate()
                self.bg.wait()

            # 2. Revert to the snapshot
            print("Reverting to snapshot...")
            self.run_remote(["vmrun", "revertToSnapshot", self.vmx_path, self.snapshot_name])
            time.sleep(5)

            # 3. Start the VM
            print("Starting VM...")
            self.bg = self.run_remote_bg(["vmrun", "start", self.vmx_path, "nogui", "&& cmd /K"])
            time.sleep(60)
        except subprocess.CalledProcessError as e:
            print("Error during environment reset:", e)
            return False

        print(f"Environment reset completed in {time.time() - start:.1f} seconds")
        return True