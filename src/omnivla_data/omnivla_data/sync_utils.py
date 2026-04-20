#!/usr/bin/env python3
"""
sync_utils.py
-------------
Provides utility classes for synchronizing asynchronous ROS messages based on timestamps.
"""

import threading

class SyncBuffer:
    """
    Stores the latest message for each key.
    get_synced() returns a dict of messages when the timestamp difference of all keys is <= slop_sec.
    """

    def __init__(self, keys: list, slop_sec: float):
        self._keys   = keys
        self._slop   = slop_sec
        self._msgs   = {k: None for k in keys}
        self._stamps = {k: 0.0  for k in keys}
        self._lock   = threading.Lock()

    def update(self, key: str, msg, stamp_sec: float):
        """Updates the buffer with the latest message and its timestamp in seconds."""
        with self._lock:
            self._msgs[key]   = msg
            self._stamps[key] = stamp_sec

    def get_synced(self):
        """
        If the timestamps of all messages are within the slop range, returns a {key: msg} dict;
        Otherwise, returns None.
        """
        with self._lock:
            # Check if we have received at least one message for every key
            if any(m is None for m in self._msgs.values()):
                return None
            
            # Check if the time difference between the oldest and newest message is within slop
            stamps = list(self._stamps.values())
            if max(stamps) - min(stamps) > self._slop:
                return None
            
            return dict(self._msgs)