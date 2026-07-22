"""Data validator checking records against Pydantic DTO models and raising ValidationError."""

import logging
from typing import List, Dict, Any
from pydantic import ValidationError as PydanticValidationError
from data_engine.models.dto import OHLCVDTO
from core.utils.exceptions import ValidationError

logger = logging.getLogger("data_engine.processors.validator")

class DataValidator:
    """Validates records using Pydantic DTOs and database constraints."""

    def validate_records(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validates all records. Raises ValidationError if a row is invalid.

        Args:
            records: Ingested raw/processed dictionaries.

        Returns:
            List of validated record dictionaries.
        """
        validated = []
        for idx, record in enumerate(records):
            try:
                # Validate with Pydantic model
                dto = OHLCVDTO(**record)
                # Keep as dictionary, but using standard types parsed by Pydantic
                validated.append(dto.model_dump() if hasattr(dto, "model_dump") else dto.dict())
            except (PydanticValidationError, ValueError) as ve:
                logger.error(f"Record validation failed at index {idx}: {record}. Error: {ve}")
                raise ValidationError(f"Data quality check failed: {ve}") from ve
                
        return validated
