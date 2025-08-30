#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
import sys
from pathlib import Path

# Create a logger instance
logger = logging.getLogger("cdaweb_downloader")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Console handler (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    ch.setFormatter(formatter)

    # File handler (writes to log file)
    log_dir = Path.cwd() / "logs"       # <--- save logs in ./logs/
    log_dir.mkdir(exist_ok=True)
    fh = logging.FileHandler(log_dir / "cdaweb_downloader.log", mode="a")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    # Add both handlers
    logger.addHandler(ch)
    logger.addHandler(fh)

    # Capture warnings too
    logging.captureWarnings(True)