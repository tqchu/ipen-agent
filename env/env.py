import logging
import numpy as np  # for vectorized state representation

from env.action import PentestAction
from env.state import State
from env.vmware import VmwareController
from metasploit.msf_rpc import MsfRpcController
from tools.exploit import Exploiter
from tools.scanners import Scanner
from tools.vulners import VulnerabilityScanner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class PenTestEnvironment:
    """
    Reinforcement Learning environment for autonomous penetration testing.
    This environment connects to real tools (Nmap, Metasploit) to perform actions and updates its state accordingly.
    It supports optional integration with a recommender system to guide action selection.
    """

    def __init__(self, target_network: str = None, msf_password: str = "truongquangchu", msf_host: str = "127.0.0.1",
                 msf_port: int = 55553, recommender=None, use_recommender: bool = False, max_hosts: int = 5):
        """
        Initialize the PenTestEnvironment.
        :param target_network: The target network or list of initial target hosts to start with (e.g., "192.168.56.0/24").
                               If None, you can still add initial actions manually.
        :param msf_password: Password for Metasploit RPC (msfrpcd).
        :param msf_host: Host running Metasploit RPC.
        :param msf_port: Port for Metasploit RPC.
        :param recommender: An optional recommender system object with a method recommend_action(state, actions).
        :param use_recommender: If True, the environment will use the recommender's suggested action on each step (overriding agent's choice).
        :param max_hosts: Maximum number of hosts to consider in state vector (for vectorization).
                          This defines the size of the observation vector for the RL agent.
        """
        # Initialize Metasploit RPC controller
        self.msf = MsfRpcController.get_instance(password=msf_password, host=msf_host, port=msf_port)  # connect to Metasploit RPC
        # Store recommender and config
        self.recommender = recommender
        self.vmware = VmwareController()
        self.executed_action = 0
        self.use_recommender = use_recommender
        # Max hosts for vector representation
        self.max_hosts = max_hosts

        # Structured state: dictionary to hold discovered information
        self.state = State()
        # Flat vector state for agent (NumPy array)
        self.vector_state = None

        # Initialize the action space (list of PentestAction objects)
        self.actions = []
        self._init_actions(target_network)
        self.target_network = target_network
        logging.info("Initial actions size: %s", len(self.actions))
        # Note: self.actions will be dynamically updated as state evolves (e.g., adding new exploit actions for discovered services)

        # Optional: track initial Metasploit exploit module list for reference
        self.exploit_modules = self.msf.list_exploit_modules()

        # Set initial observation vector
        self._update_vector_state()

    def _init_actions(self, target_network):
        """
        Populate initial actions (usually reconnaissance actions like network scanning).
        If a target network or host list is provided, create scanning actions for host discovery and port scanning.
        This runs at environment initialization or reset.
        """
        # Clear any existing actions
        self.actions = []
        # If target_network is specified, we can add a host discovery scan for that network
        if target_network:
            # Add scan port action
            discover_action = PentestAction(
                action_type="scan",
                target=target_network,
                tool=Scanner(),
                description=f"Host discovery on {target_network}"
            )
            self.actions.append(discover_action)

            # Add vulnerability scan action
            vuln_scan_action = PentestAction(
                action_type="vuln_scan", target=target_network, tool=VulnerabilityScanner(),
                description=f"Vulnerability scan on {target_network}"
            )
            self.actions.append(vuln_scan_action)

            # Add exploit actions
            exploit_modules = self.msf.get_all_exploit()
            for module in exploit_modules:
                # Create an exploit action for each module
                exploit_action = PentestAction(
                    action_type="exploit", target=target_network, module=module, tool=Exploiter(),
                    description=f"Exploit {module} against {target_network}"
                )
                self.actions.append(exploit_action)

        # If specific hosts are known upfront (target_network could be a single IP), we can also add port scan actions directly
        # (In many cases, target_network might be an IP or small range that we treat similarly.)
        # The environment can also be initialized with no targets, expecting the user/agent to supply scanning actions as needed.

    def reset(self):
        """
        Reset the environment to the initial state (start a new episode).
        This will clear discovered information and reset available actions to the initial set.
        If any sessions were open or exploit jobs running, it will attempt to clean them up.
        :return: Initial observation (vector state).
        """
        # Close any active sessions from previous episode
        for sess_id in list(self.msf.client.sessions.list.keys()):
            self.msf.close_session(sess_id)
        # Reset state
        self.state.reset()
        # Re-initialize actions (starting over with initial scan actions)
        # We assume target_network is stored or passed again; for simplicity, not storing target_network, so user can set it again if needed.
        # If needed, we could store self.target_network in __init__ and use it here.
        # For now, we'll just clear actions and keep any initial ones that were set in __init__.
        self.actions = []
        self._init_actions(self.target_network)
        # logging.info("Actions after reset: %s", self.actions)
        # Update vector state
        self._update_vector_state()
        # TODO: uncomment
        # self.vmware.reset()
        self.executed_action = 0
        return self.vector_state

    def step(self, action):
        """
        Execute one step in the environment using the given action.
        :param action: The action to perform. Can be either an index (int) referring to self.actions list, or a PentestAction object.
        :return: A tuple (next_state_vector, reward, done, info)
        """
        self.executed_action += 1
        # Determine the actual action object
        if self.recommender and self.use_recommender:
            # If recommender is enabled, get the recommended action (override agent's choice for execution)
            recommended_action = self.recommender.recommend_action(self.state, self.actions)
            if isinstance(recommended_action, PentestAction):
                actual_action = recommended_action
            else:
                # If recommender returns an index or similar, translate it to action
                try:
                    actual_action = self.actions[recommended_action]
                except Exception:
                    # If recommendation is invalid, default to using the provided action
                    actual_action = recommended_action if isinstance(recommended_action, PentestAction) else None
        else:
            # No recommender override, use the action given by the agent/user
            if isinstance(action, PentestAction):
                actual_action = action
            elif isinstance(action, int):
                actual_action = self.actions[action]  # convert index to action object
            else:
                raise ValueError("Action must be an index or PentestAction object")
        if actual_action is None:
            raise ValueError("No valid action to execute")

        logging.info("Executing action: %s", actual_action)
        info = {"action": actual_action}
        reward = 0.0
        done = False

        # Execute the action based on its type
        result = actual_action.tool.run(actual_action, self.actions, self.state)
        # dynamically change the state and actions
        if result:
            # Update the state based on the action result
            reward += result.reward
            self.actions.extend(result.future_actions)
            self.state.update(result)
            logging.debug("Action result: %s", result)


        # Update the vectorized state after performing the action
        self._update_vector_state()
        logging.info("Updated state vector: %s", self.vector_state)

        # Prepare the observation (next state) for return
        observation = self.vector_state.copy() if isinstance(self.vector_state, np.ndarray) else self.vector_state

        done = self.state.is_done()

        return observation, reward, done, info

    def _update_vector_state(self):
        """
        Update self.vector_state based on the current self.state.
        The vector is structured to a fixed length (depending on max_hosts and other considerations).
        """
        # Initialize a zero vector of a fixed length
        # We'll encode: [num_hosts, num_exploited_hosts, num_credentials, any_root_owned (0/1)] + per-host features
        num_features = 4 + self.max_hosts * 3  # for example, 4 global + 3 per host (found, exploited, num_ports)
        state_vector = np.zeros(num_features, dtype=float)
        discovered_hosts = self.state.discovered_hosts
        num_hosts = len(discovered_hosts)
        exploited_hosts = self.state.exploited_hosts
        credentials = self.state.credentials
        # Global features
        state_vector[0] = num_hosts
        state_vector[1] = len(exploited_hosts)
        state_vector[2] = len(credentials)
        # Feature: any host with root access obtained
        root_owned = 0
        for host, priv in self.state.privileges.items():
            if priv == "root":
                root_owned = 1
                break
        state_vector[3] = root_owned
        # Per-host features (for first max_hosts hosts)
        offset = 4
        for i in range(self.max_hosts):
            if i < num_hosts:
                host_ip = discovered_hosts[i]
                # Host found
                state_vector[offset] = 1
                # Host exploited
                state_vector[offset + 1] = 1 if host_ip in exploited_hosts else 0
                # Number of open ports (normalized by 10 for example)
                num_ports = len(self.state.open_ports.get(host_ip, []))
                state_vector[offset + 2] = min(num_ports, 10) / 10.0  # cap at 10 ports for normalization
            else:
                # No host data for this slot
                state_vector[offset] = 0
                state_vector[offset + 1] = 0
                state_vector[offset + 2] = 0
            offset += 3
        self.vector_state = state_vector

    def get_vector_state(self):
        """
        Get the current state as a flat vector (NumPy array). This can be used as an observation for RL algorithms.
        """
        return self.vector_state

    def get_state(self):
        """
        Get the current structured state (dictionary). This is useful for debugging or feeding into a recommender.
        """
        return self.state

    def render(self):
        """
        Optional: Render the current state in a human-readable form (print discovered hosts, services, etc.).
        """
        print("=== PenTestEnvironment State ===")
        print(f"Discovered hosts: {self.state.discovered_hosts}")
        for host in self.state.discovered_hosts:
            print(f" Host {host}:")
            ports = self.state.open_ports.get(host, [])
            for port, service in ports:
                print(f"   - Port {port}/tcp open ({service})")
            if host in self.state.vulnerabilities:
                vulns = self.state.vulnerabilities[host]
                if vulns:
                    print(f"   Vulnerabilities: {vulns}")
            if host in self.state.privileges:
                priv = self.state.privileges[host]
                print(f"   Compromise: {priv} access obtained")
        if self.state.credentials:
            print(f"Credentials found: {self.state.credentials}")
        if self.state.exploited_hosts:
            print(f"Exploited hosts: {self.state.exploited_hosts}")
        print("================================")
