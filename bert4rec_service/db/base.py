from sqlalchemy.orm import DeclarativeBase

from bert4rec_service.db.meta import meta


class Base(DeclarativeBase):
    """Base for all models."""

    metadata = meta
