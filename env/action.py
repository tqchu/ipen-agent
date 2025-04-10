from tools.base_tool import BaseTool


class PentestAction:
    """
    Representation of a single pentesting action, including what to run and against which target.
    """

    def __init__(self, action_type: str, target: str = None, port: int = None, service: str = None, module: str = None,
                 tool: BaseTool = None, description: str = ""):
        """
        Initialize an action.
        :param action_type: Type of action ("scan", "vuln_scan", "exploit", etc.).
        :param target: Target host (IP address or network range). For scanning actions, this could be a subnet.
        :param port: Target port (if applicable, e.g. for exploit or service-specific scan).
        :param service: Service name (if known) associated with the port.
        :param module: The module or exploit name to use (for Metasploit exploits or specific scanner module).
        :param tool: The external tool name if not using Metasploit (e.g., "nmap").
        :param description: Human-readable description of the action.
        """
        self.type = action_type
        self.target = target
        self.port = port
        self.service = service
        self.module = module  # Metasploit module name for exploits or scanner modules
        self.tool = tool  # Tool to execute (e.g., "nmap", "msf") if needed to differentiate
        self.params = {}  # Additional parameters for the action (e.g., scan type, flags, etc.)
        # If no description provided, generate a simple one
        if description:
            self.description = description
        else:
            if action_type == "scan":
                self.description = f"Scan target {target}" if target else "Network scan"
            elif action_type == "vuln_scan":
                self.description = f"Vulnerability scan on {target}"
            elif action_type == "exploit":
                self.description = f"Exploit {module} on {target}:{port}"
            else:
                self.description = action_type

    def __repr__(self):
        return f"<Action type={self.type}, target={self.target}, port={self.port}, module={self.module}, tool={self.tool}>"


class Result:
    """
    Representation of the result of a scanning action.
    """

    def __init__(self, reward: float, discovered_hosts: list[str] = None,
                 open_ports: dict = None, vulners: dict = None,
                 future_actions: list[PentestAction] = None,
                 status: str = None, details: dict = None,
                 hosts_shell: dict = None,
                 hosts_meterpreter: dict = None
                 ):
        """
        Initialize a scan result object.
        :param reward: Reward of the action.
        :param discovered_hosts: List of discovered host IPs.
        :param open_ports: Dictionary mapping host IPs to lists of (port, service) tuples.
        :param vulners: List of vulnerabilities found.
        :param status: Status of the action (e.g., "success", "failure").
        :param details: Additional details about the result.
        """
        self.reward = reward
        self.status = status
        self.details = details or {}
        self.discovered_hosts = discovered_hosts or []
        self.open_ports = open_ports or {}
        self.vulners = vulners or {}
        self.future_actions = future_actions or []
        self.hosts_shell = hosts_shell
        self.hosts_meterpreter = hosts_meterpreter

    def __repr__(self):
        return f"Result(reward={self.reward}, discovered_hosts={self.discovered_hosts}, " \
               f"open_ports={self.open_ports}, vulners={self.vulners})"
