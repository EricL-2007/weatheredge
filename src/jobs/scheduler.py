from apscheduler.schedulers.blocking import BlockingScheduler
from src.jobs.daily_sync import main as daily_sync

def run_job():
    print("Starting scheduled sync...")
    daily_sync()
    print("Scheduled sync finished.")

sched = BlockingScheduler(timezone="America/Chicago")
sched.add_job(run_job, trigger="cron", hour=6, minute=0, id="daily_sync")

if __name__ == "__main__":
    print("Scheduler started. Press Ctrl+C to stop.")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped.")