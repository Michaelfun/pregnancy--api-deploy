import multiprocessing

# Gunicorn configuration
bind = "0.0.0.0:10000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "gthread"
threads = 2
timeout = 120
keepalive = 5
accesslog = "-"
errorlog = "-"
loglevel = "info"
preload_app = True

# Server socket
backlog = 2048

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