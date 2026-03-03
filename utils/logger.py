"""
Logging utilities for training.
"""

import csv
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from tqdm import tqdm


class MetricsLogger:
    """Logger for training metrics."""

    def __init__(self, log_dir: str, metrics_keys: list, resume: bool = False):
        """
        Args:
            log_dir: Directory to save logs
            metrics_keys: List of metric names to log
            resume: If True, append to existing files and load history
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_keys = ['epoch'] + metrics_keys
        self.metrics_file = self.log_dir / 'metrics.csv'

        # Storage for plotting
        self.history = {key: [] for key in metrics_keys}

        if resume and self.metrics_file.exists():
            # Load existing history from metrics.csv
            with open(self.metrics_file, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    for key in metrics_keys:
                        if key in row and row[key]:
                            self.history[key].append(float(row[key]))
            # Append mode for CSV (no new header)
        else:
            # Fresh start: write CSV header
            with open(self.metrics_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=self.metrics_keys)
                writer.writeheader()

        # Console log file (append when resuming)
        self.console_log = self.log_dir / 'training_log.txt'
        self.log_file = open(self.console_log, 'a' if resume else 'w')

    def log_metrics(self, epoch: int, metrics: Dict[str, float]):
        """
        Log metrics for current epoch.

        Args:
            epoch: Epoch number
            metrics: Dictionary of metric name -> value
        """
        # Update history
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)

        # Write to CSV
        row = {'epoch': epoch}
        row.update(metrics)

        with open(self.metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.metrics_keys)
            writer.writerow(row)

    def log_message(self, message: str, print_console: bool = True):
        """
        Log a message to file and optionally console.

        Args:
            message: Message to log
            print_console: Whether to also print to console
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] {message}\n"

        self.log_file.write(log_line)
        self.log_file.flush()

        if print_console:
            print(message)

    def close(self):
        """Close log file."""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()

    def __del__(self):
        """Cleanup."""
        self.close()


class TqdmLogger:
    """Wrapper for tqdm progress bars."""

    def __init__(self, total: int, desc: str, use_tqdm: bool = True):
        """
        Args:
            total: Total number of iterations
            desc: Description for progress bar
            use_tqdm: Whether to use tqdm (otherwise just print)
        """
        self.use_tqdm = use_tqdm
        if use_tqdm:
            self.pbar = tqdm(total=total, desc=desc, ncols=100)
        else:
            self.total = total
            self.current = 0
            self.desc = desc

    def update(self, n: int = 1):
        """Update progress."""
        if self.use_tqdm:
            self.pbar.update(n)
        else:
            self.current += n

    def set_postfix(self, postfix_dict):
        """Set postfix info."""
        if self.use_tqdm:
            self.pbar.set_postfix(postfix_dict)

    def close(self):
        """Close progress bar."""
        if self.use_tqdm:
            self.pbar.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def setup_results_dir(base_dir: str, experiment_name: str) -> Path:
    """
    Create timestamped results directory.

    Args:
        base_dir: Base results directory
        experiment_name: Name of the experiment

    Returns:
        Path to results directory
    """
    # Add microseconds to prevent race conditions in parallel execution
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    results_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    results_dir.mkdir(parents=True, exist_ok=False)

    # Create subdirectories
    (results_dir / 'visualizations').mkdir(exist_ok=True)

    return results_dir
