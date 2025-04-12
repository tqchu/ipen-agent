import time
from pymetasploit3.msfrpc import MsfRpcClient, PayloadModule

from cache.metasploit_cache import MetasploitCache

def wait_for_job_completion(job_info, client):
    if job_info is not None:
        if "error" in job_info:
            return
        job_is_running = True
        while job_is_running:
            job_id = job_info["uuid"]
            results = client.jobs.info_by_uuid(job_id)
            if "error" in results:
                return
            if results["status"] == "completed":
                job_is_running = False
            else:
                time.sleep(1)

class ExploitResult:
    def __init__(self, success=False, session_id=None, session_type=None, output="", error=None):
        self.success = success
        self.session_id = session_id
        self.session_type = session_type
        self.output = output
        self.error = error

    def __repr__(self):
        return (f"ExploitResult(success={self.success}, session_id={self.session_id}, "
                f"session_type={self.session_type}, output={self.output}, error={self.error})")

class MsfRpcController:
    """
    Utility class to manage Metasploit RPC connections and actions.
    Connects to Metasploit's msfrpcd/msgrpc and provides methods for scanning and exploitation.
    """

    _instance = None

    @classmethod
    def get_instance(cls, password: str, host: str = "127.0.0.1", port: int = 55553, ssl: bool = True):
        """Singleton pattern to ensure cache is built only once."""
        if cls._instance is None:
            cls._instance = cls(password, host, port, ssl)
        return cls._instance

    def __init__(self, password: str, host: str = "127.0.0.1", port: int = 55553, ssl: bool = True):
        """
        Initialize the Metasploit RPC controller by connecting to the RPC server.
        :param password: Password for msfrpcd/msfconsole RPC (set when starting the service).
        :param host: Host where msfrpcd is running (default localhost).
        :param port: Port of the RPC service (default 55553 for msfrpcd).
        :param ssl: Whether to use SSL for the connection.
        """
        if MsfRpcClient is None:
            raise ImportError("pymetasploit3 not installed. Please install pymetasploit3 to use Metasploit RPC.")
        # Connect to the Metasploit RPC server
        self.client = MsfRpcClient(password, host=host, port=port, ssl=ssl)
        # Once connected, we can use self.client.modules, self.client.sessions, etc.
        # Example: self.client.modules.exploits, self.client.modules.auxiliary
        # Ensure the connection is authenticated
        self.cache = MetasploitCache.get_instance(self, self.list_exploit_modules())
        self.user_pass_file = "/opt/metasploit-framework/embedded/framework/data/wordlists"

        assert self.client.authenticated, "Metasploit RPC authentication failed"

    def list_exploit_modules(self) -> list:
        """
        Retrieve a list of all exploit module names available in Metasploit.
        """
        return self.client.modules.exploits

    def list_auxiliary_modules(self) -> list:
        """
        Retrieve a list of all auxiliary module names available (including scanners).
        """
        return self.client.modules.auxiliary

    def get_all_exploit(self):
        return self.cache.get_all_exploit_modules()

    def get_exploit_for_cve(self, cve):
        return self.cache.get_exploit_for_cve(cve)

    def get_auxiliary_for_cve(self, cve):
        return self.cache.get_auxiliary_for_cve(cve)

    def run_exploit(self, module_name: str, target_ip: str, target_port: int, payload: str = None) -> ExploitResult:
        """
        Launch a Metasploit exploit module against a target host/port. Returns a result dict with success status and details.
        :param module_name: Name of the exploit module (e.g. "unix/ftp/vsftpd_234_backdoor").
        :param target_ip: Target IP address to attack.
        :param target_port: Target port number of the service to exploit.
        :param payload: Optional payload to use. If None, use default or first compatible payload.
        :return: dict with keys: 'success' (bool), 'session_id' (if success), 'output' (module output text), 'error' (if any).
        """
        result = ExploitResult(success=False, session_id=None, output="", error=None)
        try:
            exploit = self.client.modules.use('exploit', module_name)
        except Exception as e:
            result.error = f"Failed to load module {module_name}: {e}"
            return result

        # Set target options
        try:
            exploit['RHOSTS'] = target_ip
            exploit["VERBOSE"] = True
            if 'RPORT' in exploit.options and target_port is not None:
                exploit['RPORT'] = target_port

            if 'USERPASS_FILE' in exploit.options:
                exploit['USERPASS_FILE'] = self.user_pass_file
            if 'ConnectTimeout' in exploit.options:
                exploit['ConnectTimeout'] = 60
            if 'WfsDelay' in exploit.options:
                exploit['WfsDelay'] = 5

            # If payload is specified, use it; otherwise use a default or first compatible payload
            chosen_payload = payload
            if not chosen_payload:
                try:
                    # If exploit has a list of compatible payloads, pick the first one
                    payloads = exploit.targetpayloads()
                    if payloads:
                        chosen_payload = PayloadModule(self.client, payloads[0])
                except Exception:
                    chosen_payload = None
            # Launch the exploit module. We'll use a console to capture output for monitoring.
            console_id = self.client.consoles.console().cid  # create a new console
            console = self.client.consoles.console(console_id)
            try:
                # Run the exploit and capture output (this will execute and block until module finishes or timeout)
                output = console.run_module_with_output(exploit, payload=chosen_payload)
                result.output = output
            except Exception as e:
                result.error = f"Error running exploit: {e}"
            # Check if a new session opened
            sessions = self.client.sessions.list
            if sessions:
                # If any session is present, assume success (for simplicity, we treat any session creation as success)
                # In a real scenario, you might compare session list before and after, or track session IDs.
                new_session_id = next(iter(sessions.keys()))
                result.success = True
                result.session_id = new_session_id
                result.session_type = self.client.sessions.list[new_session_id]["type"]
        except Exception as e:
            result.error = f"Error running exploit: {e}"
        return result

    def run_auxiliary(self, module_name: str, target_ip: str = None, target_port: int = None,
                      options: dict = None) -> dict:
        """
        Run a Metasploit auxiliary module against a target. Returns a result dict with success status and details.
        :param module_name: Name of the auxiliary module (e.g. "scanner/portscan/tcp").
        :param target_ip: Target IP address (if required by the module).
        :param target_port: Target port number (if required by the module).
        :param options: Additional options to set for the module as a dictionary.
        :return: dict with keys: 'success' (bool), 'output' (module output text), 'error' (if any), 'results' (any results).
        """
        result = {"success": False, "output": "", "error": None, "results": None}
        try:
            auxiliary = self.client.modules.use('auxiliary', module_name)
        except Exception as e:
            result["error"] = f"Failed to load auxiliary module {module_name}: {e}"
            return result

        # Set target options if provided
        if target_ip is not None and 'RHOSTS' in auxiliary.options:
            auxiliary['RHOSTS'] = target_ip
        if target_port is not None and 'RPORT' in auxiliary.options:
            auxiliary['RPORT'] = target_port

        # Set additional options
        if options:
            for key, value in options.items():
                if key in auxiliary.options:
                    auxiliary[key] = value

        # Check for missing required options
        missing_required = getattr(auxiliary, 'missing_required', [])
        if missing_required:
            result["error"] = f"Missing required options: {', '.join(missing_required)}"
            return result

        # Launch the auxiliary module using a console to capture output
        console_id = self.client.consoles.console().cid  # create a new console
        console = self.client.consoles.console(console_id)
        try:
            # Run the auxiliary module and capture output
            output = console.run_module_with_output(auxiliary)
            result["output"] = output

            # Check for errors in output
            if "[-]" in output or "error" in output.lower():
                result["error"] = "Error detected in module output"
            else:
                result["success"] = True
        except Exception as e:
            result["error"] = f"Error running auxiliary module: {e}"
        finally:
            # Clean up the console
            try:
                console.destroy()
            except:
                pass  # Ignore errors during cleanup

        return result

    def get_session(self, session_id: str):
        """
        Get a session object by ID for interaction (e.g., to run commands on a shell or meterpreter session).
        """
        try:
            return self.client.sessions.session(session_id)
        except Exception as e:
            print(f"[MsfRpcController] Error retrieving session {session_id}: {e}")
            return None

    def close_session(self, session_id: str):
        """
        Close/stop a Metasploit session by ID.
        """
        try:
            sess = self.client.sessions.session(session_id)
            sess.stop()  # terminate the session
        except Exception as e:
            print(f"[MsfRpcController] Error closing session {session_id}: {e}")

    def disconnect(self):
        """Terminate the RPC connection."""
        try:
            self.client.logout()
        except Exception:
            pass
