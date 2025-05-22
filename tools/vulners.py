import logging
from collections import defaultdict

import formatter.formatters
from env.v1.action import PentestAction, Result
from metasploit.msf_rpc import MsfRpcController
from tools.base_tool import BaseTool
from nmap import PortScanner

from tools.exploit import Exploiter
from exploitdb.exploitdb import ExploitDbResolver
from tools.exploit_db import ExploitDbRunner
from utils.rewards import REWARDS, RewardType

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

class VulnerabilityScanner(BaseTool):
    def __init__(self, use_cache_vulners = False):
        super().__init__(name="Scanner")
        self.msf = MsfRpcController.get_instance(password="truongquangchu", host="localhost", port=55553)
        self.exploit_db = ExploitDbResolver()
        self.exploit_modules = self.msf.list_exploit_modules()
        self.use_cache_vulners = use_cache_vulners

    def run(self, action: PentestAction, actions: list[PentestAction], state):
        # Vulnerability scanning (using Nmap scripts or analyzing service versions)
        host = action.target

        nm = PortScanner()
        # Use Nmap with vulners or vuln scripts to identify vulnerabilities on open ports
        args = "--script vulners -sV -T4 -Pn"

        # future_actions = []

        nm.scan(hosts=host, arguments=args)
        return self._extract_vulnerabilities_from_nmap(nm, host, state)

    def _extract_vulnerabilities_from_nmap(self, nm, host, state) -> [float, dict]:
        """
        Extract vulnerabilities from nmap scan results

        Args:
            nm: PortScanner object with scan results
            host: Target host IP
            state: Current environment state

        Returns:
            float: Reward value for discovered vulnerabilities
            dict: Dictionary of discovered vulnerabilities
        """
        reward = 0.0

        vulnerabilities = defaultdict(list)

        discovered_hosts = []
        open_ports = defaultdict(list)
        if host in nm.all_hosts():
            for port in nm[host].all_tcp():
                port_vulners = []

                service_name = nm[host]['tcp'][port]['name']

                service_info = nm[host]['tcp'][port].get('product', '')

                # Compose service description
                service_desc = service_name
                if service_info:
                    service_desc += f" ({service_info})"

                known_host = state.get_host(host)
                know_port = None
                if known_host is None:
                    if host not in discovered_hosts:
                        discovered_hosts.append(host)
                    open_ports[host].append((port, service_desc))
                else:
                    know_port = known_host.get_port(port)
                    if know_port is None:
                        open_ports[host].append((port, service_desc))

                if nm[host]['tcp'][port]['state'] == 'open':
                    script_results = nm[host]['tcp'][port].get('script', {})
                    if script_results:
                        for script_name, script_output in script_results.items():
                            vulner = None
                            cve_vulne = None
                            edb_vulne = None
                            ps_vulne = None
                            if script_name.lower().startswith("vulners"):
                                for line in script_output.splitlines():
                                    if "CVE-" in line:
                                        cve_id = line.strip().split()[0]
                                        cve_vulne = formatter.formatters.format_cve(cve_id)

                                    if "EDB-"  in line:
                                        edb_id = line.strip().split()[0]
                                        edb_vulne = formatter.formatters.format_edb_id(edb_id)

                                    if "PACKETSTORM"  in line:
                                        packetstorm_id = line.strip().split()[0]
                                        ps_vulne = formatter.formatters.format_packetstorm_id(packetstorm_id)

                                if cve_vulne:
                                    vulner = cve_vulne
                                elif edb_vulne:
                                    vulner = edb_vulne
                                elif  ps_vulne:
                                    vulner = ps_vulne
                                if vulner is None:
                                    break

                                if vulner not in port_vulners and (know_port is None or vulner not in know_port.vulnerabilities):
                                    port_vulners.append(vulner)
                                    logging.info("Discovered vulnerability %s on host %s", vulner, host)

                if len(port_vulners) > 0:
                    vulnerabilities[host].append((port, port_vulners))


        if len(vulnerabilities) > 0:
            reward += REWARDS[RewardType.VULNERABILITY]

        return Result(
            reward=reward,
            vulners=vulnerabilities,
            discovered_hosts= discovered_hosts,
            open_ports=open_ports
        )
