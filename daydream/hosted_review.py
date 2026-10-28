"""Trusted operator policy for the bounded native Pi committed-review entry point.

This is model/tool authority, not infrastructure isolation. The immutable request
adds independently validated inert Git preparation; the legacy request retains
the operator's preparation responsibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType

from daydream.backends import BackendExecutionInput
from daydream.review_budget import ReviewLimits
from daydream.review_profile import ResolvedProfile

HOSTED_PI_TOOLS = frozenset({"read", "find", "ls", "grep", "structured_output"})


class UnsupportedHostedPolicy(ValueError):
    """The requested authority cannot be established before native model work."""


@dataclass(frozen=True, kw_only=True)
class HostedReviewPolicy:
    """Operator-owned inputs; never derive these from reviewed content or ambient state.

    ``agent_dir`` is an isolated native configuration directory containing only
    approved provider/authentication configuration and inert settings. Credentials
    remain live for native refresh. ``environment`` is the complete child process
    environment, including its trusted executable PATH and credential variables.
    Host outputs and private diagnostics use disjoint operator-selected roots.
    """

    agent_dir: Path
    environment: Mapping[str, str] = field(repr=False, compare=False)
    output_dir: Path
    private_base: Path
    profile: ResolvedProfile
    limits: ReviewLimits = field(default_factory=ReviewLimits)
    trusted_prepared_source: bool = False
    backend: str = "pi"
    provider: str = "nous"
    model: str = "deepseek/deepseek-v4.1-flash-1017"

    def __post_init__(self) -> None:
        object.__setattr__(self, "environment", MappingProxyType(dict(self.environment)))

    def execution_input(self, source: Path) -> BackendExecutionInput:
        """Validate authority before source preparation and freeze native dispatch inputs."""
        if self.trusted_prepared_source is not True:
            raise UnsupportedHostedPolicy(
                "Hosted review requires trusted_prepared_source=True: the operator must prepare and audit "
                "source staging before invoking this API. Legacy requests additionally require audited "
                "Git configuration, checkout hooks and filters; immutable requests use the inert Git owner."
            )
        if self.backend != "pi":
            raise UnsupportedHostedPolicy(
                "Hosted review supports only Pi; "
                "select the supported operator policy without switching harnesses."
            )
        if any(not isinstance(value, str) or not value.strip() for value in (self.provider, self.model)):
            raise UnsupportedHostedPolicy(
                "Hosted review requires nonblank operator-selected provider and model strings")
        roots = (source, self.agent_dir, self.output_dir, self.private_base)
        if any(not path.is_absolute() or path.resolve() != path for path in roots):
            raise UnsupportedHostedPolicy(
                "Hosted source/configuration/output/private roots must be canonical absolute paths")
        if any(a.is_relative_to(b) or b.is_relative_to(a)
               for index, a in enumerate(roots) for b in roots[index + 1:]):
            raise UnsupportedHostedPolicy("Hosted source, configuration, output and private roots must be disjoint")
        if not self.agent_dir.is_dir():
            raise UnsupportedHostedPolicy(
                "Hosted agent_dir must be an existing directory; prepare isolated operator configuration before review")
        if self.output_dir.exists() or self.output_dir.is_symlink():
            raise UnsupportedHostedPolicy("Hosted output_dir must be initially absent; select a unique run destination")
        if not isinstance(self.profile, ResolvedProfile):
            raise UnsupportedHostedPolicy("Hosted review requires an explicitly resolved operator profile")
        environment = dict(self.environment)
        environment.update(HOME=str(self.agent_dir), PI_CODING_AGENT_DIR=str(self.agent_dir), PI_PROVIDER=self.provider)
        return replace(
            BackendExecutionInput.from_environment(environment, backend="pi"), pi_hosted_read_only=True,
        )
