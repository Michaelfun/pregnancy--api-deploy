#!/bin/bash

# Create necessary directories
mkdir -p logs
mkdir -p data

# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv nginx

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements-prod.txt

# Train the model if not already trained
if [ ! -f "pregnancy_vitals_model.joblib" ]; then
    echo "Training model..."
    python modal_training.py
fi

# Create Nginx configuration
sudo tee /etc/nginx/sites-available/pregnancy_vitals << EOF
server {
    listen 80;
    server_name your_domain.com;  # Replace with your domain

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }
}
EOF

# Enable Nginx configuration
sudo ln -s /etc/nginx/sites-available/pregnancy_vitals /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Create systemd service
sudo tee /etc/systemd/system/pregnancy_vitals.service << EOF
[Unit]
Description=Pregnancy Vitals Monitoring API
After=network.target

[Service]
User=ubuntu  # Replace with your user
Group=www-data
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/gunicorn -c gunicorn_config.py api:app
Restart=always

[Install]
WantedBy=multi-user.target
EOF

# Start and enable the service
sudo systemctl daemon-reload
sudo systemctl start pregnancy_vitals
sudo systemctl enable pregnancy_vitals

echo "Deployment completed! The API is now running at http://your_domain.com" 