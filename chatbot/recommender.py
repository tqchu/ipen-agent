#!/usr/bin/env python3
import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Union, List, Optional

from chatbot.qa import ModelManager
from env.v1.action import PentestAction
from env.v1.state import PenTestState
from model.enum import ShellType
from tools.exploit import Exploiter
from tools.scanners import Scanner

logger = logging.getLogger("Recommender")


class Recommender(ABC):
    """
    Abstract base class for recommenders that suggest penetration testing actions.
    Implementations can use different strategies (rule-based, ML, etc.).
    """

    def __init__(self, config: dict = None):
        """
        Initialize the recommender with optional configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        self.config = config or {}
        logger.info(f"Initialized {self.__class__.__name__} recommender")

    def recommend_action(self, state: PenTestState, available_actions: List[PentestAction]) -> Union[
        PentestAction, int]:
        """
        Template method that recommends the next action to take.
        This coordinates the recommendation process.

        Args:
            state: Current state of the penetration test
            available_actions: List of available actions to choose from

        Returns:
            Either a PentestAction object or an index into available_actions
        """
        pass


class QARecommender(Recommender):
    def __init__(self, config: dict = None):
        """
        Initialize the QA recommender with optional configuration.

        Args:
            config: Dictionary containing configuration parameters
        """
        super().__init__(config)
        self.model_manager = ModelManager()
        logger.info("QARecommender initialized")

    def recommend_action(self, state: PenTestState, available_actions: List[PentestAction]) -> (Union[int, None],
                                                                                                float):
        """
        Use the language model to recommend an action based on the current penetration test state.

        Args:
            state: Current state of the penetration test
            available_actions: List of available actions to choose from

        Returns:
            Either a PentestAction object or an index into available_actions
            Also return confidence score of the recommendation
        """
        logger.debug(f"QA Recommender analyzing state and {len(available_actions)} available actions")

        # Get human-readable state information
        state_info = state.get_readable_state()

        # Construct a question based on discovered hosts, open ports, and vulnerabilities
        question = "Based on the current penetration testing state:\n\n"

        # Add information about discovered hosts
        if not state_info:
            question += "No hosts have been discovered yet. Which initial scanning action is recommended?\n\n"
        else:
            question += f"{len(state_info)} hosts have been discovered.\n\n"

            # Add details for each host
            for host in state_info:
                host_idx = host['host_index']
                question += f"Host {state.hosts_ip[host_idx]} (Shell level: {host['shell_level']}, Privilege level: {host['privilege_level']}):\n"

                # Add port and vulnerability information
                if host['open_ports']:
                    question += "- Open ports:\n"
                    for port_info in host['open_ports']:
                        port = port_info['port']
                        vulns = port_info['vulnerabilities']
                        question += f"  - Port {port}"
                        if vulns:
                            question += f" with {len(vulns)} vulnerabilities: {', '.join(vulns)}"
                        question += "\n"
                else:
                    question += "- No open ports discovered\n"

                # Add credential information
                creds = host['credentials']
                if creds['usernames'] or creds['credentials'] or creds['tokens']:
                    question += "- Credentials:\n"
                    if creds['usernames']:
                        question += f"  - Usernames: {', '.join(creds['usernames'])}\n"
                    if creds['credentials']:
                        formatted_creds = [f"{u}:{p}" for u, p in creds['credentials']]
                        question += f"  - User:Pass: {', '.join(formatted_creds)}\n"
                    if creds['tokens']:
                        question += f"  - Tokens: {len(creds['tokens'])} found\n"

                question += "\n"

        # Add available actions information
        question += "Available actions:\n"
        for i, action in enumerate(available_actions):
            question += f"{i}. {action}\n"

        question += "\nBased on the current state, which action number would be most effective for progressing the penetration test? Respond with only the action number."

        # Get recommendation from the model
        logger.debug(f"Sending question to model: {question[:100]}...")
        try:
            # Try to extract the recommended action index
            logger.info(f"Construct question for the model: {question}")

            answer, conf = self.model_manager.get_answer(
                question,
                system_prompt="You are an expert penetration tester. Your task is to recommend the next best action based on the current state."
                              "You should answer in the format of the json, the first field is reason, the second field is actions."
                              "For 'reason' field, you should explain why you recommend these actions, for the actions, you should return a list of actions sorted by order of execution, each actions should contain its index in the list and its content, which is the same as the list of available actions I pass in."
            )

            # Try to extract the recommended action index
            logger.info(f"Model answer: {answer}")

            # Try to extract a number from the answer
            try:
                # Try to parse the JSON part of the answer
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', answer, re.DOTALL)

                if json_match:
                    json_str = json_match.group(1)
                    response_data = json.loads(json_str)

                    # Extract the first recommended action from the actions list
                    if "actions" in response_data and len(response_data["actions"]) > 0:
                        first_action = response_data["actions"][0]
                        if "index" in first_action:
                            action_idx = int(first_action["index"])

                            # Ensure the index is valid
                            if 0 <= action_idx < len(available_actions):
                                logger.info(f"Recommending action {action_idx}: {available_actions[action_idx]}")
                                return action_idx, conf
                            else:
                                logger.warning(f"Invalid action index {action_idx}, falling back to default handling")
                        else:
                            logger.warning("Action index not found in recommended action")
                    else:
                        logger.warning("No actions found in model response")
                else:
                    # Fallback to the old number extraction method
                    match = re.search(r'\b(\d+)\b', answer)
                    if match:
                        action_idx = int(match.group(1))
                        # Ensure the index is valid
                        if 0 <= action_idx < len(available_actions):
                            logger.info(f"Recommending action {action_idx}: {available_actions[action_idx]}")
                            return action_idx, conf
                        else:
                            logger.warning(f"Invalid action index {action_idx}, falling back to default handling")
                    else:
                        logger.warning(f"Could not extract action index from answer: {answer}")
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON from model response: {e}")
            except Exception as e:
                logger.error(f"Error processing model response: {e}")

        except Exception as e:
            logger.error(f"Error getting recommendation from model: {e}")

        # Fall back to parent class implementation if model recommendation fails
        return None, 0


if __name__ == "__main__":
    # Example usage
    recommender = QARecommender()

    MH, MP = 10, 10
    state = PenTestState(MH, MP)

    # Populate Host 0
    state.add_host("192.168.100.168", privilege_level=0)
    state.add_open_port("192.168.100.168", 22)
    state.add_vulnerability("192.168.100.168", 22, "CVE-2021-1111")
    state.add_open_port("192.168.100.168", 80)
    state.add_vulnerability("192.168.100.168", 80, "CVE-2021-2222")
    state.add_username("192.168.100.168", "admin")
    state.add_credential("192.168.100.168", "admin", "password123")

    # Populate Host 1
    state.add_host("192.168.100.169", privilege_level=1)
    state.add_open_port("192.168.100.169", 443)
    state.mark_exploited("192.168.100.169", ShellType.METERPRETER)
    state.add_vulnerability("192.168.100.169", 443, "CVE-2022-3333")
    state.add_vulnerability("192.168.100.169", 443, "CVE-2023-4444")
    state.add_token("192.168.100.169", "session-token-abc")

    available_actions = [PentestAction(action_type="scan", target="192.168.100.168", tool=Scanner()),
                         PentestAction(action_type="exploit", target="192.168.100.169", module="exploit_module.py")]
    action = recommender.recommend_action(state, available_actions)
    print(f"Recommended action: {action}")
