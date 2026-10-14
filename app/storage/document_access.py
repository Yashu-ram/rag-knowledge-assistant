"""Read access to personal uploads and the bundled sample library."""

SAMPLE_LIBRARY_USER_ID = "system-sample-library"


def readable_owner_ids(user_id: str) -> list[str]:
    return list(dict.fromkeys([user_id, SAMPLE_LIBRARY_USER_ID]))


def can_read_document(document, user_id: str) -> bool:
    return document is not None and document.user_id in readable_owner_ids(user_id)
