import logging

from env.vmware import VmwareController

class PenTestEnvironment:
    """Environment for autonomous penetration testing.
    Manages state of target systems and simulates attack actions.
    Integrates with VMware to reset and control VMs between episodes.
    """

    def __init__(self, target_vm):
        """
        Initialize the pentest environment.
        """
        # Connect to VMware controller for managing VMs
        self.vmware = VmwareController()
        # Define action space (list of possible actions agent can take)
        self.actions = ["SCAN_NETWORK", "SCAN_PORTS", "EXPLOIT_WEAK_SERVICE", "ESCALATE_PRIVILEGES"]
        # Internal state representation (could be any data structure: e.g., dict of discovered info)
        self.state = None
        logging.info("PenTestEnvironment initialized for VM: %s", target_vm)

    def reset(self):
        """Reset the environment to the initial state using a VM snapshot.
        Returns the initial state for a new episode.
        """
        # Reset the VM
        self.vmware.reset()

        # Reset the state to initial values
        self.state = self._get_initial_state()
        logging.info("Environment reset")
        return self.state

    def _get_initial_state(self):
        """Helper to define the initial state of the environment."""
        # Example initial state: no vulnerabilities discovered, initial recon data empty.
        initial_state = {"discovered_services": [], "credentials": [], "privilege_level": "none"}
        return initial_state

    def step(self, action_index):
        """
        Execute the given action in the environment and update state.
        :param action_index: Index of the action to perform (corresponding to self.actions).
        :return: (next_state, reward, done, info)
            - next_state: the new state after action
            - reward: reward for this action (float)
            - done: whether the episode is finished (bool)
            - info: dict with extra info (e.g., {"goal_reached": bool})
        """
        action = self.actions[action_index]
        logging.info("Executing action: %s", action)
        # Simulate or execute the action. For example:
        if action == "SCAN_NETWORK":
            # e.g., run Nmap on target VM's network to discover open ports
            # result = self.vmware.execute(self.target_vm, "nmap -Pn -p- -T4 [target_ip]")
            # Update state with discovered open ports/services (simulated here)
            self.state["discovered_services"] = ["ftp", "ssh"]  # example outcome
            reward = 0.1  # small reward for gathering info
            done = False
        elif action == "SCAN_PORTS":
            # Example of more detailed scan on a specific service
            # ... update state
            reward = 0.1
            done = False
        elif action == "EXPLOIT_WEAK_SERVICE":
            # Attempt an exploit on a known vulnerable service (e.g., misconfigured FTP)
            # If successful, update state to reflect gained access
            success = True  # (determine via simulation or tool output)
            if success:
                self.state["privilege_level"] = "user"  # gained low-level access
                reward = 1.0  # high reward for successful exploit
                done = True  # episode can end if goal is to gain access
            else:
                reward = -0.1
                done = False
        elif action == "ESCALATE_PRIVILEGES":
            # Attempt privilege escalation after initial exploit
            success = True
            if success:
                self.state["privilege_level"] = "root"
                reward = 1.0
                done = True  # reached final goal (root access)
            else:
                reward = -0.1
                done = False
        else:
            # Unknown action
            reward = -0.0
            done = False

        # Determine if goal is reached (e.g., root access obtained)
        goal_reached = (self.state.get("privilege_level") == "root")
        info = {"goal_reached": goal_reached}
        # Set next_state (could be internal state or processed observation)
        next_state = self.state
        logging.info("Action result -> State: %s, Reward: %.2f, Done: %s", next_state, reward, done)
        return next_state, reward, done, info

    def close(self):
        """Clean up environment (disconnect VMware, etc.)"""
        # self.vmware.close()
        logging.info("Environment closed and VMware connection terminated.")
