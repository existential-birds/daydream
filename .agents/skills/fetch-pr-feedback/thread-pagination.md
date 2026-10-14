# Review-thread pagination

Use explicit cursor loops so thread pagination and each thread's comment
pagination are independently complete. Preserve every page as a distinct file in
the acquisition directory. The following queries are executable first-page
examples; repeat them with `-F cursor="$NEXT_CURSOR"` for subsequent pages.

## Threads

```bash
gh api graphql -F owner="$OWNER" -F repo="$REPO" -F pr="$PR_NUMBER" \
  -f query='
  query($owner: String!, $repo: String!, $pr: Int!, $cursor: String) {
    repository(owner: $owner, name: $repo) {
      pullRequest(number: $pr) {
        id
        headRefOid
        reviewThreads(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id
            isResolved
            isOutdated
            path
            line
            startLine
            diffSide
            startDiffSide
            comments(first: 100) {
              pageInfo { hasNextPage endCursor }
              nodes {
                id
                databaseId
                url
                body
                createdAt
                updatedAt
                author { login }
              }
            }
          }
        }
      }
    }
  }' > "$FEEDBACK_DIR/thread-page-001.json"

jq -e '
  (.errors == null or (.errors | length == 0)) and
  (.data.repository.pullRequest.reviewThreads.nodes | type == "array") and
  (.data.repository.pullRequest.reviewThreads.pageInfo.hasNextPage | type == "boolean")
' "$FEEDBACK_DIR/thread-page-001.json" > /dev/null
```

After each successful page, read
`.data.repository.pullRequest.reviewThreads.pageInfo`. If `hasNextPage` is true,
require a nonempty `endCursor` different from the cursor just used. Pass it as
`cursor` on the next call and use a new page filename. Merge all thread nodes by
`id` only after the final page reports false. Retain resolution and outdated state
on each thread, including threads whose comments will later be excluded.

## Comments within a thread

For **each** collected thread whose initial `comments.pageInfo.hasNextPage` is
true, fetch its remaining comment pages using the following query. `THREAD_ID`
is that thread's node ID and `NEXT_CURSOR` is its initial comments end cursor.
`THREAD_INDEX` and `PAGE_INDEX` are unique local integer counters for filenames.

```bash
gh api graphql -f thread="$THREAD_ID" -F cursor="$NEXT_CURSOR" \
  -f query='
  query($thread: ID!, $cursor: String) {
    node(id: $thread) {
      ... on PullRequestReviewThread {
        id
        comments(first: 100, after: $cursor) {
          pageInfo { hasNextPage endCursor }
          nodes {
            id
            databaseId
            url
            body
            createdAt
            updatedAt
            author { login }
          }
        }
      }
    }
  }' > "$FEEDBACK_DIR/thread-$THREAD_INDEX-comment-page-$PAGE_INDEX.json"

jq -e --arg thread "$THREAD_ID" '
  (.errors == null or (.errors | length == 0)) and
  (.data.node.id == $thread) and
  (.data.node.comments.nodes | type == "array") and
  (.data.node.comments.pageInfo.hasNextPage | type == "boolean")
' "$FEEDBACK_DIR/thread-$THREAD_INDEX-comment-page-$PAGE_INDEX.json" > /dev/null
```

Read `.data.node.comments.pageInfo` after each page, applying the same cursor
checks. Combine the initial and subsequent comments by node `id`; associate every
`databaseId` with the parent thread's `id`, `isResolved`, and `isOutdated`.
Completion requires the last comments page of **every** thread to report
`hasNextPage: false`. Report null comment database IDs as unmatched evidence when
they cannot be joined to REST; never silently discard a page or assume a limit
of 100 is sufficient.
