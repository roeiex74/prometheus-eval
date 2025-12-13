# Agent State Update Template

This template should be appended to every agent MD file to ensure they know how to update the centralized state.

---

## State Management Protocol (MANDATORY)

### Primary State File

**Location**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json`

This is the **single source of truth** for project status. You MUST update this file when:
1. Starting a task
2. Completing a task
3. Encountering blockers
4. Updating quality metrics (if applicable)

### Your Status Section

Your status is tracked in the `agents` section under your agent name:

```json
{
  "agents": {
    "your-agent-name": {
      "role": "Your Role Description",
      "tasks_assigned": 0,
      "tasks_completed": 0,
      "tasks_in_progress": 0,
      "current_task": null,
      "status": "IDLE",
      "last_active": "ISO timestamp"
    }
  }
}
```

### Update Protocol

#### When Starting a Task

**Before you begin work**, update your status to indicate you're active:

```python
import json
from datetime import datetime

# Read current state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update your status
agent_name = 'your-agent-name'  # Replace with your actual agent name
state['agents'][agent_name]['status'] = 'ACTIVE'
state['agents'][agent_name]['current_task'] = 'Description of what you are doing'
state['agents'][agent_name]['tasks_in_progress'] += 1
state['agents'][agent_name]['last_active'] = datetime.utcnow().isoformat() + 'Z'

# Update global metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = agent_name

# Save state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

#### When Completing a Task

**After successfully finishing your work**, update your status:

```python
import json
from datetime import datetime

# Read current state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update your status
agent_name = 'your-agent-name'  # Replace with your actual agent name
state['agents'][agent_name]['status'] = 'IDLE'
state['agents'][agent_name]['current_task'] = None
state['agents'][agent_name]['tasks_completed'] += 1
state['agents'][agent_name]['tasks_in_progress'] -= 1
state['agents'][agent_name]['last_active'] = datetime.utcnow().isoformat() + 'Z'

# Update global metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = agent_name

# Save state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

#### When Encountering a Blocker

**If you cannot proceed**, update your status and notify the orchestrator:

```python
import json
from datetime import datetime

# Read current state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update your status
agent_name = 'your-agent-name'  # Replace with your actual agent name
state['agents'][agent_name]['status'] = 'BLOCKED'
state['agents'][agent_name]['current_task'] = 'Description of blocked task'
state['agents'][agent_name]['last_active'] = datetime.utcnow().isoformat() + 'Z'

# Optionally add to issues if it's a new blocker
# (Orchestrator will typically manage the issues section)

# Update global metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = agent_name

# Save state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)

# Then return control to orchestrator with explanation
```

### Validation Agents: Additional Responsibilities

If you are a **validation agent** (qa-agent, security-agent, project-architect-agent, research-agent, documentation-agent, validation-submission-agent), you should also update quality metrics:

```python
import json
from datetime import datetime

# Read current state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update quality metrics (example for QA agent)
current_phase = state['current_phase']
phase_key = current_phase.lower().replace(' ', '_')

if 'qa-agent' in agent_name:
    # Update test metrics
    state['phases'][phase_key]['quality_metrics']['test_coverage_percent'] = 75  # Example
    state['phases'][phase_key]['quality_metrics']['test_pass_rate_percent'] = 98.5
    state['phases'][phase_key]['quality_metrics']['code_quality_score'] = 82

# Update your agent status
agent_name = 'qa-agent'
state['agents'][agent_name]['status'] = 'IDLE'
state['agents'][agent_name]['current_task'] = None
state['agents'][agent_name]['tasks_completed'] += 1
state['agents'][agent_name]['last_active'] = datetime.utcnow().isoformat() + 'Z'

# Update validation progress
state['validation_framework']['stages_completed'] += 1
state['validation_framework']['last_run'] = datetime.utcnow().date().isoformat()

# Update global metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = agent_name

# Save state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

### Quick Reference

**Your Agent Name**: `{AGENT_NAME}` (replace in code above)

**Valid Status Values**:
- `IDLE` - No active task
- `ACTIVE` - Currently working
- `BLOCKED` - Cannot proceed without intervention

**When to Update**:
- ✅ Start of task → Status: ACTIVE, current_task: description
- ✅ End of task → Status: IDLE, tasks_completed++, current_task: null
- ✅ Blocker encountered → Status: BLOCKED, current_task: blocked task

**What NOT to Do**:
- ❌ Don't skip state updates
- ❌ Don't modify other agents' sections
- ❌ Don't update without reading first
- ❌ Don't forget to increment tasks_in_progress/tasks_completed

### Legacy State Manager

You may still see references to `state_manager.py`. This is the **old system**. Continue to use it if instructed by the orchestrator, but the **primary system** is now PROJECT_STATE.json.

If you use state_manager.py, the orchestrator will sync it to PROJECT_STATE.json periodically:

```bash
python src/tools/state_manager.py \
  --agent "your-agent-name" \
  --action update \
  --phase IMPLEMENTATION \
  --status IN_PROGRESS \
  --log "Starting task description" \
  --artifact "path/to/file.py"
```

### Handoff to Orchestrator

After updating state, **always** return control explicitly:

```
Task complete. State updated in PROJECT_STATE.json.
Returning control to @project-orchestrator.
```

The orchestrator will read the updated state and decide the next action.

### More Information

See `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/docs/STATE_MANAGEMENT_GUIDE.md` for complete documentation.
