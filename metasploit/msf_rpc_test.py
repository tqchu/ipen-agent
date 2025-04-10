import time
from pymetasploit3.msfrpc import MsfRpcClient

from cache.metasploit_cache import MetasploitCache
from metasploit.msf_rpc import MsfRpcController

msf = MsfRpcController.get_instance(password="truongquangchu", host="localhost", port=55553, ssl=True)

print("OK")
