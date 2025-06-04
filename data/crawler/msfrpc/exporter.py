# exporter.py

import os
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Exporter:
    """
    Handles writing processed module dicts to JSONL or chunked JSON files.
    """

    def __init__(self, out_dir: str, chunk_size: int = 0):
        """
        :param out_dir: directory where exports will go (will be created if missing)
        :param chunk_size: if >0, split into multiple files each containing `chunk_size` modules.
                           If ==0, write a single file 'all_modules.jsonl'.
        """
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        os.makedirs(out_dir, exist_ok=True)

    def export_jsonl(self, records: List[Dict[str, Any]], filename: str = "all_modules.jsonl"):
        """
        Write each record as one JSON line to <out_dir>/<filename>.
        """
        outpath = os.path.join(self.out_dir, filename)
        with open(outpath, "w", encoding="utf-8") as f:
            for rec in records:
                line = json.dumps(rec, ensure_ascii=False)
                f.write(line + "\n")
        logger.info(f"Wrote {len(records)} records to {outpath}")

    def export_chunked(self, records: List[Dict[str, Any]], prefix: str = "chunk"):
        """
        Split `records` into chunks of size self.chunk_size; write each chunk
        to <prefix>_1.jsonl, <prefix>_2.jsonl, etc.
        """
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be >0 to use export_chunked")

        total = len(records)
        num_files = (total + self.chunk_size - 1) // self.chunk_size

        for i in range(num_files):
            start = i * self.chunk_size
            end = min(start + self.chunk_size, total)
            chunk = records[start:end]
            filename = f"{prefix}_{i+1}.jsonl"
            outpath = os.path.join(self.out_dir, filename)
            with open(outpath, "w", encoding="utf-8") as f:
                for rec in chunk:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            logger.info(f"Wrote records {start}–{end} to {outpath} ({len(chunk)} modules)")

    def export_single_json(self, records: List[Dict[str, Any]], filename: str = "all_modules.json"):
        """
        If you prefer a single JSON array (not line‐delimited), use this.
        """
        outpath = os.path.join(self.out_dir, filename)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"Wrote {len(records)} records to {outpath} (as a JSON array)")
