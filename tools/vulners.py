import logging
from collections import defaultdict

import formatter.formatters
from env.action import PentestAction, Result
from metasploit.msf_rpc import MsfRpcController
from tools.base_tool import BaseTool
from nmap import PortScanner

from tools.exploit import Exploiter
from exploitdb.exploitdb import ExploitDbResolver
from tools.exploit_db import ExploitDbRunner

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

        # Parse script output for vulnerabilities
        if self.use_cache_vulners:
            # Read vulnerabilities from cache file instead of scanning
            vulnerabilities = defaultdict(list)
            try:
                with open(f"vulnerabilities{action.target}.txt", "r") as f:
                    for line in f:
                        cve = line.strip()
                        if cve:
                            vulnerabilities[host].append(cve)
                reward = len(vulnerabilities[host])  # Reward based on number of vulnerabilities
                logging.info(f"Loaded {reward} cached vulnerabilities for {host}")
            except FileNotFoundError:
                logging.warning(f"No cached vulnerabilities file found for {host}, falling back to scanning")
                reward, vulnerabilities = self._extract_vulnerabilities_from_nmap(nm, host, state)
        else:
            nm.scan(hosts=host, arguments=args)
            reward, vulnerabilities = self._extract_vulnerabilities_from_nmap(nm, host, state)

        cves = vulnerabilities.get(host, [])

        # dump the vulnerabilities.get(host, []) to a file
        with open(f"vulnerabilities{action.target}.txt", "w") as f:
            for vuln in cves:
                f.write(f"{vuln}\n")

        # for vuln in cves:
            # matching_auxiliary_modules = self.msf.get_auxiliary_for_cve(vuln)

            # for module_name in matching_auxiliary_modules:
            #     module_name = module_name['module_name']
            #     # If module matches the vulnerability (or service name), add an exploit action
            #     # Need to know port for the exploit - find from open_ports if matching service
            #     target_port = None
            #     service_desc_list = state.open_ports.get(host, [])
            #     for (p, svc) in service_desc_list:
            #         if svc.lower() in module_name.lower():
            #             target_port = p
            #             break
            #     # If not found by vuln name, we might match by service name substring
            #     if target_port is None:
            #         # Try to infer from module path (e.g., module "unix/ftp/vsftpd_234_backdoor" -> port 21)
            #         if "vsftpd" in module_name:
            #             target_port = 21
            #         elif "smb" in module_name or "445" in module_name:
            #             target_port = 445
            #     auxiliary_action = PentestAction(
            #         action_type="auxiliary", target=host, port=target_port, module=module_name, tool=AuxiliaryRunner(),
            #         description=f"Auxiliary {module_name} against {host}"
            #     )
            #     if all(act.description != auxiliary_action.description for act in actions + future_actions):
            #         future_actions.append(auxiliary_action)
            #         logging.info("Added auxiliary action: %s", auxiliary_action)

            # matching_exploit_modules = self.msf.get_exploit_for_cve(vuln)
            # for module_name in matching_exploit_modules:
            #     # If module matches the vulnerability (or service name), add an exploit action
            #     # Need to know port for the exploit - find from open_ports if matching service
            #     target_port = None
            #     service_desc_list = state.open_ports.get(host, [])
            #     for (p, svc) in service_desc_list:
            #         if svc.lower() in module_name.lower():
            #             target_port = p
            #             break
            #     # If not found by vuln name, we might match by service name substring
            #     if target_port is None:
            #         # Try to infer from module path (e.g., module "unix/ftp/vsftpd_234_backdoor" -> port 21)
            #         if "vsftpd" in module_name:
            #             target_port = 21
            #         elif "smb" in module_name or "445" in module_name:
            #             target_port = 445
            #     if target_port is None:
            #         continue
            #
            #     exploit_action = PentestAction(
            #         action_type="exploit", target=host, port=target_port, module=module_name, tool=Exploiter(),
            #         description=f"Exploit {module_name} against {host}"
            #     )
            #     if all(act.description != exploit_action.description for act in actions + future_actions):
            #         future_actions.append(exploit_action)
            #         logging.info("Added exploit action: %s", exploit_action)

            # matching_exploit_db_modules = self.exploit_db.get_exploit_paths_of_cve(vuln)
            # for exploit_module in matching_exploit_db_modules:
            #     target_port = None
            #     service_desc_list = state.open_ports.get(host, [])
            #     for (p, svc) in service_desc_list:
            #         if svc.lower() in exploit_module.metadata.exploit.lower():
            #             target_port = p
            #             break
            #
            #     exploit_action = PentestAction(
            #         action_type="exploit", target=host, port=target_port, module=exploit_module, tool=ExploitDbRunner(),
            #         description=f"Exploit {exploit_module.metadata.exploit} against {host}"
            #     )
            #     if all(act.description != exploit_action.description for act in actions + future_actions):
            #         future_actions.append(exploit_action)
            #         logging.info("Added exploit action: %s", exploit_action)


        return Result(
            reward=reward,
            # future_actions=future_actions,
            vulners=vulnerabilities
        )

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

        if host in nm.all_hosts():
            for port in nm[host].all_tcp():
                if nm[host]['tcp'][port]['state'] == 'open':
                    script_results = nm[host]['tcp'][port].get('script', {})
                    if script_results:
                        for script_name, script_output in script_results.items():
                            if script_name.lower().startswith("vulners"):
                                vulns = []
                                for line in script_output.splitlines():
                                    if "CVE-" in line:
                                        cve_id = line.strip().split()[0]
                                        vulns.append(formatter.formatters.format_cve(cve_id))

                                    if "EDB-"  in line:
                                        edb_id = line.strip().split()[0]
                                        vulns.append(formatter.formatters.format_edb_id(edb_id))

                                    if "PACKETSTORM"  in line:
                                        packetstorm_id = line.strip().split()[0]
                                        vulns.append(formatter.formatters.format_packetstorm_id(packetstorm_id))

                                for v in vulns:
                                    if v not in state.vulnerabilities.get(host, []) + vulnerabilities[host]:
                                        vulnerabilities[host].append(v)
                                        reward += 1.0
                                        logging.info("Discovered vulnerability %s on host %s", v, host)

        return reward, vulnerabilities
