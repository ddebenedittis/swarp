"""Opt-in observation sensors (lidar, ...) computed as differentiable torch ops."""

from wmas.sensors.lidar import Lidar, lidar_scan

__all__ = ["Lidar", "lidar_scan"]
