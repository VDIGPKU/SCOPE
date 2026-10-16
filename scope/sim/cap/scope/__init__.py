"""SCOPE's simulator-independent protocol and execution archive.

Importing this package does not load a simulator, model client, or robot API.
"""

from .archive import (
    ArchiveError,
    BudgetExhaustedError,
    IdentityMismatchError,
    ScopeArchive,
    StaleArchiveError,
)
from .protocol import ScopeProtocol
from .records import (
    CampaignIdentity,
    CampaignState,
    EvidenceRef,
    ExecutionOutcome,
    ExecutionRecord,
    ExecutionResult,
    ExecutionRole,
    ExecutionStatus,
    IssueKind,
    IssueRecord,
    ProgramVersion,
    SuccessWitness,
    Termination,
)

__all__ = [
    "ArchiveError",
    "BudgetExhaustedError",
    "CampaignIdentity",
    "CampaignState",
    "EvidenceRef",
    "ExecutionOutcome",
    "ExecutionRecord",
    "ExecutionResult",
    "ExecutionRole",
    "ExecutionStatus",
    "IdentityMismatchError",
    "IssueKind",
    "IssueRecord",
    "ProgramVersion",
    "ScopeArchive",
    "ScopeProtocol",
    "StaleArchiveError",
    "SuccessWitness",
    "Termination",
]
