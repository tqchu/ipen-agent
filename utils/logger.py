import logging

def setup_logging(log_file=None):
    """
    Configure logging for the project.
    Logs messages to console (and to a file if provided) with a standard format.
    """
    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    if log_file:
        # Add a file handler in addition to console
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(fh)
