"""Index bundled samples once for all authenticated accounts."""
import logging
import secrets

from fastapi import UploadFile

from app.auth.security import hash_password
from app.core.config import get_settings
from app.ingestion.pipeline import ALLOWED_TYPES, ingest_upload
from app.storage.document_access import SAMPLE_LIBRARY_USER_ID
from app.storage.models import User

logger = logging.getLogger(__name__)


async def ingest_sample_library(db):
    directory = get_settings().sample_documents_dir
    if not directory.is_dir():
        logger.warning("Sample document directory is unavailable: %s", directory)
        return
    paths = sorted(path for path in directory.rglob("*")
                   if path.is_file() and path.suffix.lower() in ALLOWED_TYPES)
    if not paths:
        return
    owner = db.get(User, SAMPLE_LIBRARY_USER_ID)
    if owner is None:
        owner = User(user_id=SAMPLE_LIBRARY_USER_ID, full_name="Shared sample library",
                     email="sample-library@system.invalid",
                     password_hash=hash_password(secrets.token_urlsafe(48)), is_active=False)
        db.add(owner)
        db.commit()
    for path in paths:
        try:
            with path.open("rb") as handle:
                # Existing indexed file hashes are reused by the ingestion pipeline.
                upload = UploadFile(filename=path.name, file=handle)
                document = await ingest_upload(db, owner, upload)
                logger.info("Sample document ready: %s (%s)", path.name, document.document_id)
        except Exception:
            db.rollback()
            logger.exception("Could not index sample document: %s", path.name)
