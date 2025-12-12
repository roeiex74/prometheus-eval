import os
import json
import argparse
import time
from typing import Dict, Any, Optional

STATE_DIR = "agent_states"

class StateManager:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.state_file = os.path.join(STATE_DIR, f"{agent_name}_state.json")
        self._ensure_dir()

    def _ensure_dir(self):
        if not os.path.exists(STATE_DIR):
            os.makedirs(STATE_DIR)

    def load_state(self) -> Dict[str, Any]:
        """Loads the current state of the agent."""
        if not os.path.exists(self.state_file):
            return self._default_state()
        
        try:
            with open(self.state_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return self._default_state()

    def _default_state(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "last_updated": None,
            "phase": "INIT",  # INIT, VALIDATION, EXECUTION, COMPLETE
            "status": "IDLE", # IDLE, IN_PROGRESS, SUCCESS, FAILED
            "validation_report": None,
            "artifacts_created": [],
            "logs": []
        }

    def update_state(self, phase: str = None, status: str = None, 
                     validation_report: str = None, artifact: str = None, 
                     log_message: str = None):
        """Updates the agent state atomically."""
        current_state = self.load_state()
        
        current_state["last_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if phase:
            current_state["phase"] = phase
        if status:
            current_state["status"] = status
        if validation_report:
            current_state["validation_report"] = validation_report
        if artifact:
            if "artifacts_created" not in current_state:
                current_state["artifacts_created"] = []
            if artifact not in current_state["artifacts_created"]:
                current_state["artifacts_created"].append(artifact)
        if log_message:
            if "logs" not in current_state:
                current_state["logs"] = []
            current_state["logs"].append(f"[{current_state['last_updated']}] {log_message}")

        with open(self.state_file, 'w', encoding='utf-8') as f:
            json.dump(current_state, f, indent=4)
        
        print(f"State updated for {self.agent_name}: {self.state_file}")

    def get_state_summary(self) -> str:
        state = self.load_state()
        return json.dumps(state, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage Agent Context & State")
    parser.add_argument("--agent", required=True, help="Name of the agent")
    parser.add_argument("--action", choices=["load", "update"], default="load", help="Action to perform")
    
    # Update arguments
    parser.add_argument("--phase", help="Current operational phase")
    parser.add_argument("--status", help="Current status")
    parser.add_argument("--report", help="Path to validation report")
    parser.add_argument("--artifact", help="Path to created artifact")
    parser.add_argument("--log", help="Log message to append")

    args = parser.parse_args()
    
    manager = StateManager(args.agent)
    
    if args.action == "load":
        print(manager.get_state_summary())
    elif args.action == "update":
        manager.update_state(
            phase=args.phase,
            status=args.status,
            validation_report=args.report,
            artifact=args.artifact,
            log_message=args.log
        )
