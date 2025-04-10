import json
import logging
import os
import pickle
from collections import defaultdict

class GeneralCache:
    def __init__(self, cache_file, cache_func=None):
        self.cache_file = cache_file
        self.cache_func = cache_func
        self.data = {}
        self._load_or_build_cache()

    def _load_or_build_cache(self):
        if os.path.exists(self.cache_file):
            self._load_cache()
        else:
            self.data = self.cache_func()
            self._save_cache(self.data)

    def _load_cache(self):
        """Load the cache from file."""
        try:
            logging.info(f"Loading cache from {self.cache_file}...")
            with open(self.cache_file, 'rb') as f:
                self.data = pickle.load(f)
            logging.info(f"Cache loaded with {len(self.data)}")
        except Exception as e:
            logging.error(f"Error loading cache file: {e}")
            self.cache_func()

    def _save_cache(self, cached_data):
        """Save the cache to a file."""
        try:
            logging.info(f"Saving exploit module cache to {self.cache_file}...")
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cached_data, f)
            logging.info("Cache saved successfully")
        except Exception as e:
            logging.error(f"Error saving cache file: {e}")
