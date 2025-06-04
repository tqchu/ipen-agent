# crawler.py

import os
import json
import logging
from typing import Dict

from metasploit.msf_rpc import MsfRpcController

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class MsfCrawler:
    """
    Uses MsfClient to crawl all exploits and store raw JSON data.
    """

    def __init__(self, client: MsfRpcController, save_dir: str = None):
        """
        :param client: an initialized MsfClient instance
        :param save_dir: optional directory to persist raw JSON per-module
        """
        self.msf_client = client
        self.save_dir = save_dir
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    def crawl_all(self) -> Dict[str, dict]:
        """
        Fetch raw JSON info for every exploit module and return a dict:
            { module_path: raw_json_dict }
        If `save_dir` was provided, also write each raw JSON to
            <save_dir>/<module_path.replace('/', '_')>.json
        """
        modules = self.msf_client.list_exploit_modules()
        all_data = {}

        for idx, mod_path in enumerate(modules, start=1):
            logger.info(f"[{idx}/{len(modules)}] Fetching info for {mod_path}")
            raw = self.msf_client.get_module_info(mod_path)
            if raw:
                all_data[mod_path] = raw

                if self.save_dir:
                    # sanitize filename: replace "/" with "__"
                    filename = mod_path.strip("/").replace("/", "__") + ".json"
                    outpath = os.path.join(self.save_dir, filename)
                    with open(outpath, "w", encoding="utf-8") as f:
                        json.dump(raw, f, indent=2)
            else:
                logger.warning(f"No data returned for {mod_path}")

        return all_data
