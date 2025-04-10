from env.action import PentestAction, Result
from tools.base_tool import BaseTool
from nmap import PortScanner
import logging

from tools.vulners import VulnerabilityScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class Scanner(BaseTool):
    def __init__(self):
        super().__init__(name="Scanner")
        self.nm = PortScanner()

    def run(self,  action:PentestAction, actions: list[PentestAction], state) -> Result:
        """Perform a scanning action and return scan results without modifying input state."""
        self.logger.info(f"Scanning with action: {action.description}, params: {action.params}")

        nm = self.nm
        reward = 0
        target = action.target

        # Initialize result containers
        discovered_hosts = []
        open_ports = {}
        future_actions = []

        if "/" in target or target.endswith(".0"):  # Network scan
            # Host discovery (ping scan)
            args = "-sn -PE -PA21,22,80,445 -T4"
            self.nm.scan(hosts=target, arguments=args)
            hosts_list = nm.all_hosts()

            for h in hosts_list:
                if nm[h].state() == "up":
                    if h not in state.discovered_hosts:
                        discovered_hosts.append(h)
                        open_ports[h] = []

            logging.info(f"Discovered hosts: {discovered_hosts}")

            # Reward for discovering hosts
            new_hosts_found = len(discovered_hosts)
            reward = 0.5 * new_hosts_found

            # Generate port scan actions for newly discovered hosts
            for h in discovered_hosts:
                portscan_action = PentestAction(
                    action_type="scan", target=h, tool=Scanner(),
                    description=f"Port scan on {h}"
                )
                # Add if not already in the action list
                if all(act.description != portscan_action.description for act in actions):
                    future_actions.append(portscan_action)

                    logging.info(f"Added new port scan action for host: {h}")

        else:  # Single host port scan
            args = ""  # Default scan arguments
            nm.scan(hosts=target, arguments=args)
            host = target

            if host not in state.discovered_hosts:
                discovered_hosts.append(host)
                open_ports[host] = []

            if host in nm.all_hosts():
                # Process open ports and services
                for proto in nm[host].all_protocols():
                    if proto == 'tcp':
                        ports = nm[host]['tcp'].keys()
                        for port in ports:
                            port_state = nm[host]['tcp'][port]['state']
                            if port_state == "open":
                                service_name = nm[host]['tcp'][port]['name']
                                service_info = nm[host]['tcp'][port].get('product', '')

                                # Compose service description
                                service_desc = service_name
                                if service_info:
                                    service_desc += f" ({service_info})"

                                # Check if this is a new port
                                known_ports = [p for (p, _) in state.open_ports.get(host, [])]
                                if port not in known_ports:
                                    if host not in open_ports:
                                        open_ports[host] = []
                                    open_ports[host].append((port, service_desc))
                                    reward += 0.1

                # Create vulnerability scan action
                vuln_scan_action = PentestAction(
                    action_type="vuln_scan", target=host, tool=VulnerabilityScanner(),
                    description=f"Vulnerability scan on {host}"
                )
                if all(act.description != vuln_scan_action.description for act in actions):
                    future_actions.append(vuln_scan_action)
                    logging.info(f"Added vuln_scan_action for host: {host}")

        # Return a proper ScanResult
        return Result(
            reward=reward,
            discovered_hosts=discovered_hosts,
            open_ports=open_ports,
            status="success",
            details={"tool": "nmap", "target": target},
            future_actions=future_actions,
        )