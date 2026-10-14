# GitHub operations

Use these patterns only after the target, evaluation, and write authorization in `SKILL.md` are established. Run commands sequentially; stop the affected operation after any failure. The identifiers below come from checked API responses, never reviewer text.

## Read a complete conversation

Reuse the raw snapshot acquired by [fetch-pr-feedback](../../fetch-pr-feedback/SKILL.md), then refresh the conversation before publishing. Its private temporary directory preserves REST review comments, issue comments, reviews, and every GraphQL thread/comment page. Follow [thread-pagination.md](../../fetch-pr-feedback/thread-pagination.md) for the complete outer thread and per-thread comment cursor loops and error checks; an outer loop alone cannot exhaust nested comments.

- REST collections: `gh api --paginate --slurp` produces an array of page arrays. Flatten with `jq 'add // []'` while retaining the full original records and stable numeric IDs.
- Preserve each comment's global `id`, numeric `databaseId`, URL, author, full body, and timestamps; join REST `id` to GraphQL `databaseId`. Record the root REST comment ID (`in_reply_to_id == null`) for inline replies. The complete REST record supplies reply-parent IDs that are not inferred from matching file/line.
- Reconcile REST and GraphQL identities. Missing, duplicate, or conflicting mappings leave resolution unavailable; do not guess a thread or root from its position.

Check each GraphQL response for an empty/missing `errors` array **and** the expected non-null `data` shape. This check is in addition to the `gh` process exit status. Keep raw failure evidence and report incomplete coverage rather than interpreting a partial response as an empty queue.

## Post an inline reply safely

Create a private `TASK_TMP` directory with `mktemp -d` and write the exact intended Markdown to `response.md` through a file-writing tool. If using a heredoc, use a quoted delimiter so Markdown's backticks and dollar signs remain literal. Do not assemble a shell command by interpolating that Markdown.

Use the evaluated root review-comment's numeric `ROOT_COMMENT_ID`:

```bash
jq -n --rawfile body "$TASK_TMP/response.md" '{body: $body}' > "$TASK_TMP/reply.json"
gh api --method POST \
  "repos/$OWNER/$REPO/pulls/$PR_NUMBER/comments/$ROOT_COMMENT_ID/replies" \
  --input "$TASK_TMP/reply.json" > "$TASK_TMP/reply-result.json"
jq -e --argjson root "$ROOT_COMMENT_ID" --arg login "$CURRENT_USER" \
  '(.id | type == "number") and (.html_url | type == "string") and
   .in_reply_to_id == $root and .user.login == $login' \
  "$TASK_TMP/reply-result.json"
```

Persist the returned reply ID/URL against the reviewed comment ID and intended body before attempting resolution. If `gh` fails or the response cannot be validated, fetch the entire conversation to determine whether the reply was actually created. A timeout can follow a successful write. Retry only after confirming the equivalent reply is absent; otherwise reuse its confirmed ID, or report uncertainty.

For an issue comment or a top-level review body, GitHub has no inline thread-reply endpoint. An authorized response can be posted as a PR issue comment with a link to the original item via `repos/$OWNER/$REPO/issues/$PR_NUMBER/comments`, using the same JSON body-file pattern. Validate its returned ID/URL and author, and report thread resolution as not applicable.

## Resolve a verified eligible thread

Use a static GraphQL query and serialized variables so arbitrary IDs or content never become executable shell fragments. `THREAD_ID` must be the current checked mapping for the evaluated comment, and the successful reply must already be confirmed.

```bash
jq -n --arg thread "$THREAD_ID" \
  '{query: "mutation($thread: ID!) { resolveReviewThread(input: {threadId: $thread}) { thread { id isResolved } } }",
    variables: {thread: $thread}}' > "$TASK_TMP/resolve.json"
gh api graphql --input "$TASK_TMP/resolve.json" > "$TASK_TMP/resolve-result.json"
jq -e --arg thread "$THREAD_ID" \
  '((.errors // []) | length == 0) and
   .data.resolveReviewThread.thread.id == $thread and
   .data.resolveReviewThread.thread.isResolved == true' \
  "$TASK_TMP/resolve-result.json"
```

Re-read that thread by node ID and confirm `isResolved`. If resolution fails or is ambiguous, retain the reply's success and reconcile remote state before retrying the mutation. Do not create a second reply just because resolution failed. Clean up only the temporary directory owned by this invocation when its evidence is no longer needed.
