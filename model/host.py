from model.enum import ShellType, Privilege


from model.enum import ShellType, Privilege

class Port:
    def __init__(self, port: int, vulnerabilities: list[str]):
        self.port = port
        self.vulnerabilities = vulnerabilities

class Host:
    """Class representing a host in the penetration test environment."""

    def __init__(self, shell_level: ShellType = ShellType.NONE, privilege: Privilege = Privilege.NONE):
        self.open_ports = []
        self.usernames = []
        self.credentials = []
        self.tokens = []
        self.shell_level = shell_level
        self.privilege_level = privilege

    def get_port(self, port_number):
        """
        Get a port by its number.
        :param port_number: Port number to search for.
        :return: Port object if found, None otherwise.
        """
        for port in self.open_ports:
            if port.port == port_number:
                return port
        return None
