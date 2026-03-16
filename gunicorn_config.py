import os
import multiprocessing

# Server socket
port = os.environ.get("PORT", "8000")
bind = f"0.0.0.0:{port}"
backlog = 2048

# Worker processes
workers = max(2, multiprocessing.cpu_count() * 2 + 1)
worker_class = 'sync'
worker_connections = 1000
timeout = 120
keepalive = 2

# Logging
# Log to stdout/stderr (Render-friendly). File paths can fail if dirs don't exist.
accesslog = '-'
errorlog = '-'
loglevel = 'info'

# Process naming
proc_name = 'pregnancy_vitals_api'

# Server mechanics
daemon = False
pidfile = None
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None 
