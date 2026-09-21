"""URL credential shapes shared by live redaction and archive publication.

Keep this module independent of trajectory and archive lifecycles: recorded
text must be scrubbed with the same shapes the publication gate recognizes.
"""

import re

_CREDENTIAL_QUERY_KEYS = frozenset(
    {"token", "access_token", "key", "secret", "password", "credential"}
)

_URL_CREDENTIAL_PATTERN = re.compile(r'''(https?://)([^:@/\s"`]+):([^@/\s"`]+)@''', re.IGNORECASE)
_TOKEN_ONLY_USERINFO_PATTERN = re.compile(r'''(https?://)[^@/\s"`]+@''', re.IGNORECASE)
# Also recognizes user:password in database URLs with a host port. Plain
# SSH login names (git@host:path) carry no password and remain unchanged.
# The boundary avoids quadratic retries inside long non-URL text; quote
# delimiters keep redaction from consuming surrounding serialized JSON.
_SCP_USERINFO_PATTERN = re.compile(
    r'''(?<![^\s@/:"'`])([^\s@/:"'`][^\s@/:"`]*:[^\s@/:"`]+@)([^/\s@:"`]+:)(?=[^\s])''',
    re.IGNORECASE,
)
_QUERY_CREDENTIAL_PATTERN = re.compile(
    r"([?&])(" + "|".join(sorted(_CREDENTIAL_QUERY_KEYS)) + r''')=[^&\s"`#]+''',
    re.IGNORECASE,
)
