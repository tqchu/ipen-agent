# main.py

import argparse
import logging
import os

from metasploit.msf_rpc import MsfRpcController
from crawler import MsfCrawler
from processor import process_bulk
from exporter import Exporter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Crawl msfrpcd, process module metadata, and export for LLM ingestion."
    )
    parser.add_argument("--rpc-user", required=True, help="msfrpcd RPC username (e.g., 'msf')")
    parser.add_argument("--rpc-pass", required=True, help="msfrpcd RPC password")
    parser.add_argument("--rpc-host", default="127.0.0.1", help="msfrpcd host")
    parser.add_argument("--rpc-port", default=55553, type=int, help="msfrpcd port")
    parser.add_argument("--ssl", action="store_true", help="Use SSL to connect to msfrpcd")
    parser.add_argument(
        "--save-raw-dir",
        default=None,
        help="If set, save each module's raw JSON to this directory",
    )
    parser.add_argument(
        "--export-dir",
        default="exports",
        help="Directory to write processed module JSON(L) to",
    )
    parser.add_argument(
        "--chunk-size",
        default=0,
        type=int,
        help="If >0, split output into chunks of <chunk-size> modules",
    )
    return parser.parse_args()

def main():
    # args = parse_args()
    rpc_pass = "truongquangchu"  # args.rpc_user

    rpc_host = "192.168.100.216"

    # 1. Connect to msfrpcd
    client = MsfRpcController(
        password=rpc_pass,
        host=rpc_host,
        port=55553,
        ssl=True
    )

    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(current_dir, "data")

    # 2. Crawl all exploit modules (raw JSON)
    crawler = MsfCrawler(client, save_dir=target_dir)
    raw_data = crawler.crawl_all()

    # 3. Process raw JSON into normalized schema
    processed_records = process_bulk(raw_data)

    chunk_size = 0
    # 4. Export processed data
    exporter = Exporter(out_dir=target_dir, chunk_size=chunk_size)
    if chunk_size > 0:
        exporter.export_chunked(processed_records, prefix="msf_module_chunk")
    else:
        exporter.export_jsonl(processed_records)

    logger.info("Done. All modules crawled, processed, and exported.")

if __name__ == "__main__":
    main()
