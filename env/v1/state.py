from typing import Optional

from env.v1.action import Result
from model.enum import ShellType, Privilege
from model.host import Host, Port
from pentest_server.event import EventType
from utils.logger import QueueLogger


class PenTestState:
    def __init__(self, max_hosts, max_ports):
        self.max_hosts = max_hosts
        self.max_ports = max_ports
        # Initialize each host entry
        self.hosts_ip = list()  # list of host IPs
        self.hosts = [None] * max_hosts
        for i in range(max_hosts):
            self.hosts[i] = Host()

    def get_host(self, host_ip) -> Optional[Host]:
        if host_ip not in self.hosts_ip:
            return None

        return self.hosts[self.hosts_ip.index(host_ip)]

    def reset(self):
        """Reset the state to initial empty values."""
        for i in range(self.max_hosts):
            self.hosts[i] = Host()

    def update(self, result: Result, q_logger:QueueLogger):
        """Update the state with new scan results."""
        for _, host_ip in enumerate(result.discovered_hosts):
            self.add_host(host_ip)

            q_logger.log(
                EventType.HOST_DISCOVERED,
                f"Discovered host: {host_ip}"
            )

        for host_ip in result.open_ports:
            for port, service_desc in result.open_ports[host_ip]:
                self.add_open_port(host_ip, port)
                q_logger.log(
                    EventType.PORT_OPEN,
                    f"Port open on {host_ip}: {port}/{service_desc}"
                )

        for host_ip in result.vulners:
            for port, vulns in result.vulners[host_ip]:
                for vuln in vulns:
                    self.add_vulnerability(host_ip, port, vuln)

                    q_logger.log(
                        EventType.VULN_FOUND,
                        (f"Vulnerability on {host_ip}:{port} – "
                         f"{vuln.id} ({vuln.severity}, CVSS {vuln.cvss_score}): "
                         f"{vuln.description}")
                    )

        if result.hosts_shell:
            for host_ip, shell_type in result.hosts_shell.items():
                self.mark_exploited(host_ip, shell_type)

                q_logger.log(
                    EventType.SHELL_FOUND,
                    f"Shell opened on {host_ip}"
                )

        if result.hosts_meterpreter:
            for host_ip, shell_type in result.hosts_meterpreter.items():
                self.mark_exploited(host_ip, shell_type)

                q_logger.log(
                    EventType.METERPRETER_FOUND,
                    f"Meterpreter session on {host_ip}"
                )

        if result.usernames:
            for host_ip, usernames in result.usernames.items():
                for username in usernames:
                    self.add_username(host_ip, username)

                    q_logger.log(
                        EventType.USERNAME_FOUND,
                        f"Discovered username on {host_ip}: {username}"
                    )

        if result.credentials:
            for host_ip, credentials in result.credentials.items():
                for username, password in credentials:
                    self.add_credential(host_ip, username, password)

                    q_logger.log(
                        EventType.CREDENTIAL_FOUND,
                        f"Credential for {host_ip}: {username}/{password}"
                    )

        if result.tokens:
            for host_ip, tokens in result.tokens.items():
                for token in tokens:
                    self.add_token(host_ip, token)

                    q_logger.log(
                        EventType.TOKEN_FOUND,
                        f"Token captured on {host_ip}: {token}"
                    )

        if result.privileges:
            for host_ip, level in result.privileges.items():
                self.set_privilege(host_ip, level)

                q_logger.log(
                    EventType.PRIVILEGE_SET,
                    f"Privilege level set on {host_ip}: {level}"
                )

    def add_host(self, host_ip, privilege_level=0, shell_level=0) -> Optional[int]:
        """Add a discovered host if not already known, return its index."""
        if host_ip in self.hosts_ip:
            return self.hosts_ip.index(host_ip)

        self.hosts[len(self.hosts_ip)] = Host(shell_level=shell_level, privilege=privilege_level)

        self.hosts_ip.append(host_ip)

        return len(self.hosts_ip) - 1

    def mark_exploited(self, host_ip, shell_type: ShellType):
        """Mark a host as exploited with a shell type (0 = none, 1 = shell, 2 = meterpreter)."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        self.hosts[host_index].shell_level = shell_type

    def add_open_port(self, host_ip, port_number: int):
        """Record an open port on a host (with no vulnerabilities initially)."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        host = self.hosts[host_index]
        # Avoid duplicates
        for p in host.open_ports:
            if p.port == port_number:
                return
        host.open_ports.append(Port(port_number, []))
        # Sort ports by number for consistency (optional)
        host.open_ports.sort(key=lambda x: x.port)

    def add_vulnerability(self, host_ip, port_number: int, vulnerability_id: str):
        """Add a discovered vulnerability (by ID) to a host's open port."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        host = self.hosts[host_index]
        # Find or create the port entry
        for p in host.open_ports:
            if p.port == port_number:
                if vulnerability_id not in p.vulnerabilities:
                    p.vulnerabilities.append(vulnerability_id)
                return
        # If port not present, add it first
        host.open_ports.append(Port(port_number, [vulnerability_id]))
        host.open_ports.sort(key=lambda x: x.port)

    def add_username(self, host_ip, username: str):
        """Add a discovered username for a host."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        host = self.hosts[host_index]
        if username not in host.usernames:
            host.usernames.append(username)

    def add_credential(self, host_ip, username: str, password: str):
        """Add a discovered username:password credential pair for a host."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        host = self.hosts[host_index]
        cred = (username, password)
        if cred not in host.credentials:
            host.credentials.append(cred)

    def add_token(self, host_ip, token: str):
        """Add a discovered token for a host."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        host = self.hosts[host_index]
        if token not in host.tokens:
            host.tokens.append(token)

    def set_privilege(self, host_ip, level: Privilege):
        """Update the privilege level for a host (0 = none, 1 = user, 2 = root)."""
        if host_ip not in self.hosts_ip:
            raise IndexError("Host is not recognized")
        host_index = self.hosts_ip.index(host_ip)

        if level not in (0, 1, 2):
            raise ValueError("Privilege level must be 0, 1, or 2")
        self.hosts[host_index].privilege_level = level

    def to_vector(self):
        """Encode the current state into a fixed-size vector (numpy array)."""
        import numpy as np
        MH, MP = self.max_hosts, self.max_ports
        open_ports_features = MP * 2
        features_per_host = open_ports_features + 5  # MPxMP for vulns, 3 for creds/tokens, 1 for priv
        vec = np.zeros(MH * features_per_host, dtype=np.int32)
        for i in range(MH):
            host = self.hosts[i]
            # Vulnerability matrix for this host
            open_ports = np.zeros(open_ports_features, dtype=np.int32)
            # Fill in vulnerability counts for each open port (one port per row)
            for j, port_entry in enumerate(host.open_ports[:MP]):
                vuln_count = len(port_entry.vulnerabilities)
                open_ports[j] = port_entry.port  # store port number in first column
                open_ports[MP + j] = vuln_count  # store count in first column

            # Insert flattened matrix into the vector
            offset = i * features_per_host
            offset_after_open_ports = offset + open_ports_features
            vec[offset: offset_after_open_ports] = open_ports.flatten()
            # Insert credential counts and privilege
            base = offset_after_open_ports
            vec[base] = len(host.usernames)
            vec[base + 1] = len(host.credentials)
            vec[base + 2] = len(host.tokens)
            vec[base + 3] = host.shell_level
            vec[base + 4] = host.privilege_level
        return vec

    def is_done(self):
        for i in range(self.max_hosts):
            if self.hosts[i].shell_level >0:
                return True

        return False

    @classmethod
    def from_vector(cls, vector, max_hosts, max_ports):
        """Decode a vector back into a PenTestState (with placeholder data)."""
        import numpy as np
        MH, MP = max_hosts, max_ports
        open_ports_features = MP * 2

        features_per_host = open_ports_features + 5
        vec = np.array(vector, dtype=np.int32)
        if vec.size != MH * features_per_host:
            raise ValueError("Vector length does not match expected state size")
        state = cls(MH, MP)
        for i in range(MH):
            offset = i * features_per_host
            # Extract and reshape this host's vulnerability matrix
            offset_after_open_ports = offset + open_ports_features
            open_ports = vec[offset: offset_after_open_ports].reshape(open_ports_features)
            host_info = Host()
            # Determine open ports from vulnerability matrix rows
            for j in range(MP):
                port_number = open_ports[j]
                if port_number == 0:
                    # No more open ports (assuming ports are listed contiguously from row 0)
                    break
                # Use a placeholder port identifier since actual port number is unknown
                vulnerabilities = []
                # Create placeholder vulnerability IDs (since we only know the count)
                vulne_count = open_ports[MP + j]

                for k in range(vulne_count):
                    vulnerabilities.append(f"VULN-{k + 1}")

                host_info.open_ports.append(Port(port_number, vulnerabilities))
            # Decode credential counts and privilege
            base = offset_after_open_ports
            num_usernames = int(vec[base])
            num_credentials = int(vec[base + 1])
            num_tokens = int(vec[base + 2])
            shell_level = int(vec[base + 3])
            priv_level = int(vec[base + 4])
            host_info.shell_level = shell_level
            host_info.privilege_level = priv_level
            # Populate placeholder credentials based on counts
            for u in range(num_usernames):
                host_info.usernames.append(f"user{u + 1}")
            for c in range(num_credentials):
                host_info.credentials.append((f"user{c + 1}", f"pass{c + 1}"))
            for t in range(num_tokens):
                host_info.tokens.append(f"token{t + 1}")
            # If host has no information at all, treat it as undiscovered (skip adding details)
            if (len(host_info.open_ports) == 0 and num_usernames == 0 and
                    num_credentials == 0 and num_tokens == 0 and priv_level == 0 and shell_level == 0):
                continue  # leave as default (no data)
            state.hosts[i] = host_info
        return state

    def get_readable_state(self):
        """Get a simplified human-readable snapshot of the state (excluding completely undiscovered hosts)."""
        readable = []
        for idx, host in enumerate(self.hosts):
            if host is None:
                continue
            # Skip hosts with no info (undiscovered)
            if (not host.open_ports and not host.usernames and
                    not host.credentials and not host.tokens and host.privilege_level == 0 and host.shell_level == 0):
                continue
            host_entry = {
                'host_index': idx,
                'shell_level': host.shell_level,
                'privilege_level': host.privilege_level,
                'open_ports': [],
                'credentials': {
                    'usernames': list(host.usernames),
                    'credentials': list(host.credentials),
                    'tokens': list(host.tokens)
                }
            }
            for port in host.open_ports:
                host_entry['open_ports'].append({
                    'port': port.port,
                    'vulnerabilities': list(port.vulnerabilities)
                })
            readable.append(host_entry)
        return readable
