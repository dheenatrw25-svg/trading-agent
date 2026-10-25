import schedule
import time
import datetime
from market_predictor import run_predictor

def job():
    print(f"\n--- Waking up AI Predictor at {datetime.datetime.now().strftime('%H:%M:%S')} ---")
    try:
        run_predictor("BTC-USD") 
    except Exception as e:
        print(f"Error during prediction: {e}")

# Run the predictor immediately once
job()

# Then run it automatically every 60 minutes
schedule.every(60).minutes.do(job)

print("Starting Auto-Predictor Background Loop...")
print("Leave this terminal open. Press Ctrl+C to stop.")

while True:
    schedule.run_pending()
    time.sleep(1)
