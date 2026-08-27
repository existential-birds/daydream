"""The review-thread GraphQL queries may only request fields GitHub's schema defines.

The minimal subset here covers exactly the types the two queries touch
(PullRequestReviewThread, the comment node, Actor). It is the enforcement point
for "requesting a nonexistent field fails CI, never in production".
"""
from tests.harness import github_schema as gs


def test_unknown_query_fields_flags_invented_fields() -> None:
    # A query requesting fields GitHub's schema does not define must be flagged.
    bad = """
    query X($o:String!,$n:String!,$p:Int!) {
      repository(owner:$o,name:$n) { pullRequest(number:$p) { reviewThreads(first:50) {
        nodes { id isResolved isResolvedBy side startSide
                comments(first:100) { nodes { author { login type isBot } } } }
      } } }
    }
    """
    assert {"side", "startSide", "isResolvedBy", "isBot", "type"} <= gs.unknown_query_fields(bad)


def test_unknown_query_fields_accepts_aliased_real_fields() -> None:
    # The fixed projection aliases real fields to the consumer keys; the real
    # field names (diffSide/startDiffSide/__typename) are all schema-defined.
    good = """
    query Test($o:String!, $n:String!, $p:Int) {
      repository(owner:$o,name:$n) { pullRequest(number:$p) { reviewThreads(first:50) {
        pageInfo { hasNextPage endCursor }
        nodes { id isResolved isOutdated subjectType path line originalLine
                side: diffSide startSide: startDiffSide
                comments(first:100) { nodes { id databaseId body
                  author { login type: __typename } createdAt updatedAt url replyTo { id } } } }
      } } }
    }
    """
    assert gs.unknown_query_fields(good) == set()
