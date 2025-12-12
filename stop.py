import os

def stop_auto_run():
    """Create a STOP_AUTO file to signal predict.py to stop gracefully."""
    with open("STOP_AUTO", "w") as f:
        f.write("STOP")
    print("🛑 Sent STOP signal to auto-runner.")
    print("The process will stop after the current deduction is finished.")

if __name__ == "__main__":
    stop_auto_run()
