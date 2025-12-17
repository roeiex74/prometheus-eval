"""
Prompt Variator Module

Systematic variations of prompts for comparative analysis.

Implemented variators:
- BaselineVariator: Simple prompt wrapper (control group)
- FewShotVariator: 1-3 example demonstrations
- ChainOfThoughtVariator: Step-by-step reasoning prompts
- CoTPlusVariator: CoT with majority voting (self-consistency)
"""

from src.variator.base import BaseVariator
from src.variator.baseline import BaselineVariator
from src.variator.few_shot import FewShotVariator
from src.variator.cot import ChainOfThoughtVariator
from src.variator.cot_plus import CoTPlusVariator

__all__ = [
    "BaseVariator",
    "BaselineVariator",
    "FewShotVariator",
    "ChainOfThoughtVariator",
    "CoTPlusVariator",
]
