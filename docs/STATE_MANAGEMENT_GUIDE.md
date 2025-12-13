# State Management Guide for Prometheus-Eval

## Overview

The Prometheus-Eval project uses a **centralized state management system** to track progress, coordinate agents, and maintain synchronization between the master plan and actual implementation.

## Single Source of Truth

**File**: `/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json`

This JSON file serves as the **single source of truth** for:
- Current phase and completion status
- Task assignments and progress
- Agent status and availability
- Quality metrics and test results
- Issue tracking and priorities
- Timeline and milestones

## State Update Protocol

### Who Updates the State?

**Primary Responsibility**: `project-orchestrator` agent

**Secondary Updates**: All agents should update their specific section when completing tasks

### When to Update

1. **Project Orchestrator**:
   - At start of every invocation (read and compare to PLAN-LATEST.md)
   - After any agent completes a task
   - Before dispatching a new agent
   - At phase gates

2. **Individual Agents**:
   - When accepting a task (update status to "IN_PROGRESS")
   - When completing a task (update status to "COMPLETE")
   - When encountering blockers (update status to "BLOCKED")
   - When updating quality metrics (test coverage, code quality, etc.)

### How to Update

#### Option 1: Manual JSON Update (Orchestrator)

The orchestrator should directly read, modify, and write PROJECT_STATE.json:

```python
import json
from datetime import datetime

# Read current state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update state
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = 'project-orchestrator'
state['agents']['system-architect-agent']['status'] = 'ACTIVE'
state['agents']['system-architect-agent']['current_task'] = 'Fix ISSUE-001'

# Write updated state
with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

#### Option 2: Agent Status Update (Individual Agents)

Each agent should update their status in the state file when completing work:

```python
import json
from datetime import datetime

def update_agent_status(agent_name, status, current_task=None):
    with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'r') as f:
        state = json.load(f)

    state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
    state['updated_by'] = agent_name
    state['agents'][agent_name]['status'] = status
    state['agents'][agent_name]['current_task'] = current_task
    state['agents'][agent_name]['last_active'] = datetime.utcnow().isoformat() + 'Z'

    if status == 'COMPLETE':
        state['agents'][agent_name]['tasks_completed'] += 1
        state['agents'][agent_name]['tasks_in_progress'] -= 1

    with open('/Users/roeirahamim/Documents/MSC/LLM_Agents/ex6/PROJECT_STATE.json', 'w') as f:
        json.dump(state, f, indent=2)

# Example usage
update_agent_status('metric-mathematician-agent', 'COMPLETE', None)
```

#### Option 3: Using state_manager.py (Legacy)

The existing state_manager.py tool can still be used for backward compatibility:

```bash
python src/tools/state_manager.py \
  --agent "system-architect-agent" \
  --action update \
  --phase IMPLEMENTATION \
  --status IN_PROGRESS \
  --log "Starting ISSUE-001 fix" \
  --artifact "setup.py"
```

**Note**: This creates individual state files in `agent_states/` directory. The orchestrator should periodically sync these to PROJECT_STATE.json.

## State Structure

### Top-Level Keys

```json
{
  "project_name": "Project identifier",
  "current_phase": "Current phase name",
  "phase_status": "Phase completion status",
  "last_updated": "ISO 8601 timestamp",
  "updated_by": "Agent who made the update",
  "phases": { /* Phase details */ },
  "agents": { /* Agent status */ },
  "issues": { /* Issue tracking */ },
  "metrics": { /* Quality metrics */ },
  "timeline": { /* Schedule tracking */ },
  "risks": { /* Risk management */ },
  "validation_framework": { /* Validation progress */ },
  "documentation": { /* Documentation status */ }
}
```

### Phase Structure

Each phase contains:
- `status`: NOT_STARTED | IN_PROGRESS | CONDITIONAL_PASS | PASS | FAIL
- `completion_percentage`: 0-100
- `tasks`: Implementation, validation, fixes breakdown
- `quality_metrics`: Test coverage, code quality, security, etc.
- `gate_decision`: PASS | CONDITIONAL_PASS | FAIL
- `research_deliverable`: Academic output status

### Agent Structure

Each agent contains:
- `role`: Agent's primary responsibility
- `tasks_assigned`: Total tasks assigned
- `tasks_completed`: Total tasks completed
- `tasks_in_progress`: Current active tasks
- `current_task`: Description of active task
- `status`: IDLE | ACTIVE | BLOCKED
- `last_active`: Timestamp of last activity

### Issue Structure

Issues categorized by severity (critical, high, medium, low):
- `id`: Unique identifier
- `title`: Short description
- `severity`: CRITICAL | HIGH | MEDIUM | LOW
- `assigned_to`: Agent responsible
- `status`: OPEN | IN_PROGRESS | BLOCKED | RESOLVED
- `effort_hours`: Estimated effort
- `created`: Creation date
- `due_date`: Deadline

## Orchestrator Workflow

### On Every Invocation

```
1. Read PLAN-LATEST.md (master plan)
2. Read PROJECT_STATE.json (current state)
3. Compare planned vs actual:
   - Which tasks are complete?
   - Which tasks are in progress?
   - Which tasks are blocked?
   - Are we on schedule?
4. Identify next actions based on gaps
5. Dispatch appropriate agents
6. Update PROJECT_STATE.json with changes
```

### Phase Gate Review

```
1. Check phase completion percentage
2. Verify all critical tasks complete
3. Review quality metrics vs targets:
   - Test coverage ≥70%
   - Code quality ≥85
   - Security risk = LOW
   - PRD compliance = 100%
4. Review open issues (critical = 0, high = 0)
5. Make gate decision: PASS | CONDITIONAL_PASS | FAIL
6. Update state with decision
7. If PASS: Advance to next phase
8. If CONDITIONAL_PASS: List conditions to address
9. If FAIL: Trigger fix loop
```

### Task Completion Flow

```
Agent completes task
    ↓
Agent updates PROJECT_STATE.json (status = COMPLETE)
    ↓
Orchestrator reads state
    ↓
Orchestrator validates completion
    ↓
If validation passes:
    - Mark task complete in state
    - Update quality metrics
    - Identify next task
    - Dispatch next agent
If validation fails:
    - Update issue tracker
    - Assign fix to agent
    - Set status to BLOCKED
```

## Best Practices

### For Orchestrator

1. **Always read state first**: Never assume state without checking
2. **Update after every change**: Keep state current
3. **Atomic updates**: Read → Modify → Write as quickly as possible
4. **Timestamp everything**: Track when changes were made
5. **Log who made changes**: Record `updated_by` field
6. **Validate before updating**: Ensure data consistency

### For Individual Agents

1. **Update when starting work**: Set status to IN_PROGRESS
2. **Update when completing work**: Set status to COMPLETE
3. **Report blockers immediately**: Set status to BLOCKED if stuck
4. **Be specific in current_task**: Describe exactly what you're doing
5. **Update metrics when relevant**: If you affect test coverage, update it

### For All Agents

1. **Read before write**: Always read current state first
2. **Preserve existing data**: Don't overwrite unrelated fields
3. **Use ISO 8601 timestamps**: Format: `2025-12-13T01:30:00Z`
4. **Keep updates small**: Only change what's necessary
5. **Verify after update**: Read back to confirm changes saved

## Integration with PLAN-LATEST.md

The PROJECT_STATE.json should always reflect progress toward PLAN-LATEST.md:

- **PLAN-LATEST.md**: What *should* happen (the plan)
- **PROJECT_STATE.json**: What *has* happened (the reality)

The orchestrator's job is to:
1. Compare the two
2. Identify gaps
3. Take action to close gaps
4. Keep state synchronized with plan

## Version Control

**DO COMMIT**: PROJECT_STATE.json should be committed to Git
- Provides historical record of progress
- Enables rollback if needed
- Tracks project evolution over time

**Commit Frequency**:
- After completing each major task
- At end of each work session
- Before and after phase gates

## Troubleshooting

### State File Corrupted

```bash
# Restore from Git
git checkout HEAD -- PROJECT_STATE.json

# Or regenerate from validation reports
# (Orchestrator should be able to reconstruct state)
```

### State Out of Sync

```bash
# Orchestrator should:
1. Read all validation reports in docs/
2. Read agent_states/*.json files
3. Reconstruct accurate PROJECT_STATE.json
4. Commit corrected state
```

### Multiple Concurrent Updates

- **Avoid**: Only orchestrator should update during active operations
- **If needed**: Use file locking or atomic operations
- **Resolution**: Last write wins (orchestrator has final authority)

## Examples

### Example 1: Agent Completes Metric Implementation

```python
# metric-mathematician-agent completes ROUGE metric

import json
from datetime import datetime

with open('PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update agent status
state['agents']['metric-mathematician-agent']['status'] = 'COMPLETE'
state['agents']['metric-mathematician-agent']['current_task'] = None
state['agents']['metric-mathematician-agent']['tasks_completed'] += 1
state['agents']['metric-mathematician-agent']['last_active'] = datetime.utcnow().isoformat() + 'Z'

# Update phase tasks
state['phases']['phase_2']['tasks']['week_6']['tasks'].remove('ROUGE metric implementation')
state['phases']['phase_2']['tasks']['week_6']['completed'] = ['ROUGE metric implementation']

# Update state metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = 'metric-mathematician-agent'

with open('PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

### Example 2: Orchestrator Dispatches New Agent

```python
# orchestrator assigns fix to system-architect-agent

import json
from datetime import datetime

with open('PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

# Update agent status
state['agents']['system-architect-agent']['status'] = 'ACTIVE'
state['agents']['system-architect-agent']['current_task'] = 'Fix ISSUE-001: Create setup.py'
state['agents']['system-architect-agent']['tasks_assigned'] += 1
state['agents']['system-architect-agent']['tasks_in_progress'] += 1

# Update issue status
for issue in state['issues']['critical']['list']:
    if issue['id'] == 'ISSUE-001':
        issue['status'] = 'IN_PROGRESS'

# Update state metadata
state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = 'project-orchestrator'

with open('PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

### Example 3: Phase Gate Review

```python
# orchestrator performs Phase 1 gate review

import json
from datetime import datetime

with open('PROJECT_STATE.json', 'r') as f:
    state = json.load(f)

phase = state['phases']['phase_1']

# Check completion
if phase['completion_percentage'] >= 100:
    # Check quality metrics
    metrics_ok = (
        phase['quality_metrics']['test_coverage_percent'] >= 70 and
        phase['quality_metrics']['code_quality_score'] >= 85 and
        phase['quality_metrics']['prd_compliance_percent'] == 100
    )

    # Check no critical issues
    no_critical = state['issues']['critical']['count'] == 0

    if metrics_ok and no_critical:
        phase['gate_decision'] = 'PASS'
        phase['status'] = 'PASS'
        phase['actual_end_date'] = datetime.utcnow().date().isoformat()

        # Advance to Phase 2
        state['current_phase'] = 'Phase 2'
        state['phases']['phase_2']['status'] = 'IN_PROGRESS'
        state['phases']['phase_2']['start_date'] = datetime.utcnow().date().isoformat()
    else:
        phase['gate_decision'] = 'CONDITIONAL_PASS'
        phase['gate_conditions'] = [
            'Achieve 70% test coverage',
            'Resolve all critical issues'
        ]

state['last_updated'] = datetime.utcnow().isoformat() + 'Z'
state['updated_by'] = 'project-orchestrator'

with open('PROJECT_STATE.json', 'w') as f:
    json.dump(state, f, indent=2)
```

## Summary

The centralized state management system provides:
- **Single source of truth** for project status
- **Clear coordination** between agents
- **Progress tracking** against master plan
- **Quality gates** enforcement
- **Historical record** via Git commits

All agents must participate in keeping PROJECT_STATE.json accurate and current for the system to function effectively.
