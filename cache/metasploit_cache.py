import logging
import os
import pickle
import subprocess
from collections import defaultdict
import re
from typing import Optional

# configure info logging
logging.basicConfig(level=logging.INFO)

class MetasploitCache:
    """Class to cache Metasploit modules by CVE."""

    _instance = None

    # EXPLOIT_CACHE_FILE = "/home/truongchu/Academic/Graduation_Thesis/Project/AI/ipen-agent/cache/data/msf_exploit_module_cache.pkl"
    EXPLOIT_CACHE_FILE = "data/msf_all_exploit_module_cache.pkl"
    AUXILIARY_CACHE_FILE = "data/msf_auxiliary_module_cache.pkl"

    @classmethod
    def get_instance(cls, msf, exploit_modules, should_use_small_set = True):
        """Singleton pattern to ensure cache is built only once."""
        if cls._instance is None:
            cls._instance = cls(msf, exploit_modules, should_use_small_set)
        return cls._instance

    def __init__(self, msf, exploit_modules, should_use_small_set = True):
        self.msf = msf
        self.exploit_modules = exploit_modules
        self.cve_to_exploit_module_map = {}
        self.cve_to_auxiliary_module_map = {}
        self._load_or_build_cache()
        self.should_use_small_set = should_use_small_set
        self.all_exploit_modules = self._init_all_exploits()

    def _init_all_exploits(self):
        all_exploits = ['unix/ftp/vsftpd_234_backdoor']

        for _, modules in self.cve_to_exploit_module_map.items():
            for module in modules:
                if self.should_use_small_set:
                    if not (module.startswith(("linux", "unix"))) or any(
                            keyword in module for keyword in ("http", "webapp", "local", "misc")):
                        continue

                if module not in all_exploits and module != 'unix/ftp/vsftpd_234_backdoor':
                    all_exploits.append(module)

                if len(all_exploits) == 10:
                    return all_exploits



    def _load_or_build_cache(self):
        """Load cache from file if it exists, otherwise build and save it."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(current_dir, self.EXPLOIT_CACHE_FILE)):
            self._load_cache()
        else:
            self._build_ovsdb_cache()
            self._save_cache()

    def _load_cache(self):
        """Load the cache from file."""

        current_dir = os.path.dirname(os.path.abspath(__file__))
        exploit_file = os.path.join(current_dir, self.EXPLOIT_CACHE_FILE)
        auxiliary_file = os.path.join(current_dir, self.AUXILIARY_CACHE_FILE)

        try:
            logging.info(f"Loading module cache from {self.EXPLOIT_CACHE_FILE}...")
            with open(exploit_file, 'rb') as f:
                self.cve_to_exploit_module_map = pickle.load(f)
            logging.info(f"Cache loaded with {len(self.cve_to_exploit_module_map)} CVE entries")

            with open(auxiliary_file, 'rb') as f:
                self.cve_to_auxiliary_module_map = pickle.load(f)
            logging.info(f"Cache loaded with {len(self.cve_to_auxiliary_module_map)} CVE entries")
        except Exception as e:
            logging.error(f"Error loading cache file: {e}")
            # If loading fails, rebuild the cache
            self._build_ovsdb_cache()
            self._save_cache()

    def _build_ovsdb_cache(self):
        """Build the exploit_cache mapping CVEs to exploit modules."""
        exploit_cache = defaultdict(list)
        logging.info("Building exploit module exploit_cache...")

        # find searchsploit EDB to OSVDB to CVE to cache CVE to exploit module
        # if not found, map from EDB ID; PACKETSTORM ID; to exploit module

        for module_name in self.exploit_modules:
            exploit_obj = self.msf.client.modules.use('exploit', module_name)
            refs = getattr(exploit_obj, "_info", {}).get("references", [])

            for ref in refs:
                if len(ref) == 0:
                    continue

                ref_str = '-'.join([str(part) for part in ref]).upper()

                if ref[0] == 'CVE':
                    cve_id = ref_str
                    exploit_cache[cve_id].append(module_name)
                    logging.info(f"Direct CVE mapping: {cve_id} -> {module_name}")

                if ref[0] == 'EDB':
                    edb_id = ref_str
                    exploit_cache[edb_id].append(module_name)
                    logging.info(f"Direct EDB mapping: {edb_id} -> {module_name}")

                    cve_id = self.find_cve_from_edb(edb_id)
                    if cve_id is not None:
                        exploit_cache[cve_id].append(module_name)
                        logging.info(f"Indirect CVE mapping: {edb_id} -> {cve_id} -> {module_name}")

                if ref[0] == "OSVDB":
                    edb_num_id = self.find_edb_from_ovsdb(ref_str)

                    if edb_num_id is not None:
                        edb_id = f"EDB-{edb_num_id}"
                        exploit_cache[edb_id].append(module_name)
                        logging.info(f"Indirect EDB mapping: {ref_str} -> {edb_id} -> {module_name}")

                        cve_id = self.find_cve_from_edb(edb_num_id)
                        if cve_id is not None:
                            exploit_cache[cve_id].append(module_name)
                            logging.info(
                                f"Double indirect CVE mapping: {ref_str} -> {edb_id} -> {cve_id} -> {module_name}")

                if ref[0] == "PACKETSTORM":
                    packet_storm_id = ref_str
                    exploit_cache[packet_storm_id].append(module_name)
                    logging.info(f"Direct PACKETSTORM mapping: {packet_storm_id} -> {module_name}")

        self.cve_to_exploit_module_map = dict(exploit_cache)  # Convert defaultdict to regular dict for serialization

        logging.info(f"Exploit cache built with {len(exploit_cache)} CVE entries")

    def find_edb_from_ovsdb(self, ovsdb_id) -> Optional[str]:
        """Find exploit database ID from OSVDB ID using searchsploit."""
        cmd = [
            "/snap/bin/searchsploit",
            "--cve",
            ovsdb_id
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

        if proc.returncode != 0:
            logging.error(f"Failed to run searchsploit command for {ovsdb_id}")
            return None

        data = proc.stdout

        if not data:
            return None

        # Extract the numeric ID from paths like "unix/remote/17491.rb"
        path_match = re.search(r'(\d+)\.[a-z]+', data)
        if path_match:
            return path_match.group(1)  # Return just the numeric ID

        return None

    def find_cve_from_edb(self, edbId) -> Optional[str]:
        cmd = [
            "/snap/bin/searchsploit",
            "--path",
            edbId
        ]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20)

        if proc.returncode != 0:
            logging.error("Failed to run searchsploit command")
            return None
        data = proc.stdout

        if not data:
            return None

        # First check if there's a CVE ID directly in the Codes section
        if "Codes:" in data:
            codes_line = [line for line in data.split('\n') if "Codes:" in line]
            if codes_line:
                codes = codes_line[0].split("Codes:")[1].strip()
                # Look for CVE pattern
                cve_match = re.search(r'CVE-\d{4}-\d+', codes)
                if cve_match:
                    return cve_match.group(0)

        # Check if CVE appears anywhere else in the output
        cve_match = re.search(r'CVE-\d{4}-\d+', data)
        if cve_match:
            return cve_match.group(0)

    def _build_cache(self):
        """Build the exploit_cache mapping CVEs to exploit modules."""
        exploit_cache = defaultdict(list)
        logging.info("Building exploit module exploit_cache...")

        for module_name in self.exploit_modules:
            exploit_obj = self.msf.client.modules.use('exploit', module_name)
            refs = getattr(exploit_obj, "_info", {}).get("references", [])

            for ref in refs:
                ref_str = '-'.join([str(part) for part in ref]).upper()
                if 'CVE-' in ref_str:
                    cve_id = ref_str.split('CVE-')[1].split('-')[0:2]
                    cve_key = f"CVE-{'-'.join(cve_id)}"
                    exploit_cache[cve_key].append({
                        'module_name': module_name,
                        'refs': refs
                    })

        auxiliary_cache = defaultdict(list)
        logging.info("Building auxiliary module exploit_cache...")
        for module_name in self.msf.list_auxiliary_modules():
            auxiliary_obj = self.msf.client.modules.use('auxiliary', module_name)
            refs = getattr(auxiliary_obj, "_info", {}).get("references", [])

            for ref in refs:
                ref_str = '-'.join([str(part) for part in ref]).upper()
                if 'CVE-' in ref_str:
                    cve_id = ref_str.split('CVE-')[1].split('-')[0:2]
                    cve_key = f"CVE-{'-'.join(cve_id)}"
                    auxiliary_cache[cve_key].append({
                        'module_name': module_name,
                        'refs': refs
                    })

        self.cve_to_exploit_module_map = dict(exploit_cache)  # Convert defaultdict to regular dict for serialization
        self.cve_to_auxiliary_module_map = dict(auxiliary_cache)  # Convert defaultdict to regular dict for serialization

        logging.info(f"Exploit cache built with {len(exploit_cache)} CVE entries")
        logging.info(f"Auxiliary cache built with {len(auxiliary_cache)} CVE entries")

    def _save_cache(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        exploit_file = os.path.join(current_dir, self.EXPLOIT_CACHE_FILE)
        auxiliary_file = os.path.join(current_dir, self.AUXILIARY_CACHE_FILE)

        """Save the cache to a file."""
        try:
            logging.info(f"Saving exploit module cache to {self.EXPLOIT_CACHE_FILE}...")
            with open(exploit_file, 'wb') as f:
                pickle.dump(self.cve_to_exploit_module_map, f)
            logging.info("Cache saved successfully")
        except Exception as e:
            logging.error(f"Error saving cache file: {e}")

        try:
            logging.info(f"Saving auxiliary module cache to {self.AUXILIARY_CACHE_FILE}...")
            with open(auxiliary_file, 'wb') as f:
                pickle.dump(self.cve_to_auxiliary_module_map, f)
            logging.info("Cache saved successfully")
        except Exception as e:
            logging.error(f"Error saving cache file: {e}")

    def get_all_exploit_modules(self):
        """Get all exploit modules."""
        return self.all_exploit_modules

    def get_exploit_for_cve(self, cve):
        """Get modules matching a CVE."""
        return self.cve_to_exploit_module_map.get(cve, [])

    def get_auxiliary_for_cve(self, cve):
        """Get modules matching a CVE."""
        return self.cve_to_auxiliary_module_map.get(cve, [])

    def update_cache(self):
        """Force rebuilding and saving the cache."""
        self._build_cache()
        self._save_cache()