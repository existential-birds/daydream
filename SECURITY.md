# Security policy

## Supported versions

Security fixes target the latest published Daydream release and ongoing development
on the default branch. Older releases have no guaranteed security backports; upgrade
to the latest release before checking whether a problem persists. Report suspected
vulnerabilities in any version so maintainers can assess their impact.

## Report a vulnerability privately

Use [GitHub private vulnerability reporting](https://github.com/existential-birds/daydream/security/advisories/new).
You need a GitHub account. The repository's **Security → Advisories → Report a
vulnerability** page reaches the same private channel.

Do not publish vulnerabilities, exploit details, credentials, or other secrets in
public issues, pull requests, discussions, or logs. If a credential was exposed,
revoke or rotate it immediately; deleting a message does not revoke access.

Include the following in the private report:

- Affected release or commit, backend, operating system, and relevant configuration.
- A description of the vulnerability, affected component, and security impact.
- Minimal reproduction steps or a proof of concept using synthetic data and dummy
  credentials. Redact tokens, private source, personal data, and sensitive paths.
- Required privileges, attacker-controlled inputs, and any known mitigations.
- Whether the issue is already public or exploited, and your preferred credit and
  contact details for coordination.

## Response and coordinated disclosure

Maintainers aim to acknowledge reports within seven calendar days and provide an
initial assessment or progress update within fourteen calendar days. These are
best-effort targets for a volunteer project, not guaranteed response or fix times.
Follow up in the same private report if those targets pass without a response.

Maintainers will coordinate reproduction, affected versions, mitigation or a fix,
and disclosure timing with the reporter. Keep details private while that work is
underway. When a fix or mitigation is available, maintainers aim to publish a
GitHub security advisory with affected versions, upgrade guidance, and agreed
reporter credit; request a CVE when appropriate. There is no fixed embargo or
guaranteed bounty. Urgent exploitation may require earlier coordinated disclosure.
