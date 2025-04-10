from env.action import Result


class State:
    """
    Represents the state of the penetration testing environment.
    Encapsulates all discovered information and current state of the pentest.
    """

    def __init__(self):
        """Initialize an empty state."""
        self.discovered_hosts = []  # list of discovered host IPs
        self.open_ports = {}  # dict: host -> list of (port, service) tuples open
        self.vulnerabilities = {}  # dict: host -> list of identified vulnerabilities
        self.credentials = []  # list of credentials obtained
        self.exploited_hosts = []  # list of hosts successfully exploited
        self.privileges = {}  # dict: host -> privilege level obtained ("user", "root", etc.)
        self.hosts_shell = {}
        self.hosts_meterpreter = {}

    def reset(self):
        """Reset the state to initial empty values."""
        self.discovered_hosts = []
        self.open_ports = {}
        self.vulnerabilities = {}
        self.credentials = []
        self.exploited_hosts = []
        self.privileges = {}
        self.hosts_shell = {}
        self.hosts_meterpreter = {}

    def update(self, result: Result):
        self.discovered_hosts.extend(result.discovered_hosts)
        self.open_ports.update(result.open_ports)
        self.vulnerabilities.update(result.vulners)
        if result.hosts_shell:
            self.hosts_shell.update(result.hosts_shell)
        if result.hosts_meterpreter:
            self.hosts_meterpreter.update(result.hosts_meterpreter)

        if result.hosts_shell or result.hosts_meterpreter:
            for host in result.hosts_shell:
                if host not in self.exploited_hosts:
                    self.exploited_hosts.append(host)
            for host in result.hosts_meterpreter:
                if host not in self.exploited_hosts:
                    self.exploited_hosts.append(host)

    def add_host(self, host_ip):
        """Add a discovered host if not already known."""
        if host_ip not in self.discovered_hosts:
            self.discovered_hosts.append(host_ip)
            self.open_ports[host_ip] = []
            self.vulnerabilities[host_ip] = []
            return True
        return False

    def add_port(self, host_ip, port, service_desc):
        """Add an open port with service information to a host."""
        if host_ip not in self.discovered_hosts:
            self.add_host(host_ip)

        # Check if this port is already known
        for p, _ in self.open_ports[host_ip]:
            if p == port:
                return False

        self.open_ports[host_ip].append((port, service_desc))
        return True

    def add_vulnerability(self, host_ip, vuln_info):
        """Add a vulnerability to a host."""
        if host_ip not in self.vulnerabilities:
            self.vulnerabilities[host_ip] = []

        if vuln_info not in self.vulnerabilities[host_ip]:
            self.vulnerabilities[host_ip].append(vuln_info)
            return True
        return False

    def add_credential(self, cred):
        """Add discovered credentials."""
        if cred not in self.credentials:
            self.credentials.append(cred)
            return True
        return False

    def mark_as_exploited(self, host_ip, privilege_level="user"):
        """Mark a host as successfully exploited with a certain privilege level."""
        if host_ip not in self.exploited_hosts:
            self.exploited_hosts.append(host_ip)
        self.privileges[host_ip] = privilege_level
        return True

    def has_root_access(self):
        """Check if root access has been obtained on any host."""
        for host, priv in self.privileges.items():
            if priv == "root":
                return True
        return False

    def is_done(self):
        if self.hosts_shell or self.hosts_meterpreter:
            return True

        return False

    def __str__(self):
        """String representation of the state."""
        result = "=== Penetration Test State ===\n"
        result += f"Discovered hosts: {len(self.discovered_hosts)}\n"
        for host in self.discovered_hosts:
            result += f" Host {host}:\n"
            for port, service in self.open_ports.get(host, []):
                result += f"   - Port {port}/tcp open ({service})\n"
            if host in self.vulnerabilities and self.vulnerabilities[host]:
                result += f"   Vulnerabilities: {self.vulnerabilities[host]}\n"
            if host in self.privileges:
                result += f"   Compromise: {self.privileges[host]} access obtained\n"
        if self.credentials:
            result += f"Credentials found: {self.credentials}\n"

        if self.exploited_hosts:
            result += f"Exploited hosts: {self.exploited_hosts}\n"

        if  self.hosts_shell:
            result += f"Shell hosts: {self.hosts_shell}\n"

        if  self.hosts_meterpreter:
            result += f"Meterpreter hosts: {self.hosts_meterpreter}\n"
        return result