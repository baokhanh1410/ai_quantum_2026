"""Core exceptions for the AI Quantum 2026 Pipeline."""

class ConfigurationError(Exception):
    """Configuration-related errors."""
    pass

class DatabaseError(Exception):
    """Database-related errors."""
    pass

class DatabaseReadError(DatabaseError):
    """Errors occurring during database read operations."""
    pass

class DatabaseWriteError(DatabaseError):
    """Errors occurring during database write operations."""
    pass

class APIConnectionError(Exception):
    """Errors connecting to external APIs (e.g. TradingView, VNStock, Yahoo Finance)."""
    pass

class ValidationError(Exception):
    """Errors when data validation fails."""
    pass

class DataEngineError(Exception):
    """General error for the Data Engine."""
    pass

class FeatureEngineError(Exception):
    """General error for the Feature Engine."""
    pass

class DataAlignmentError(FeatureEngineError):
    """Errors when aligning or merging data sources fails."""
    pass

class FeatureCalculationError(FeatureEngineError):
    """Errors during calculation of technical or macro indicators."""
    pass

class ScalerNotFoundError(FeatureEngineError):
    """Errors when a required ML scaler object is not found."""
    pass
