import sys 
import logging
import os

# Set up logging
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

log_dir = "logs"
log_file = os.path.join(log_dir, "logging.log")
os.makedirs(log_dir, exist_ok=True)  # Changed mkdir to makedirs for safety

logging.basicConfig(
    level=logging.INFO, 
    format=logging_str, 
    handlers=[
        logging.FileHandler(log_file),  # Corrected log_filepath to log_file
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("datasciencelogger")
