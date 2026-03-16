import multiprocessing

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = 'sync'
worker_connections = 1000
timeout = 30
keepalive = 2

# Logging
accesslog = 'logs/access.log'
errorlog = 'logs/error.log'
loglevel = 'info'

# Process naming
proc_name = 'pregnancy_vitals_api'

# Server mechanics
daemon = False
pidfile = 'logs/gunicorn.pid'
umask = 0o007
user = None
group = None
tmp_upload_dir = None

# SSL
keyfile = None
certfile = None 