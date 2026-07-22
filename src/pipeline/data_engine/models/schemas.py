"""SQLAlchemy models for MySQL tables conforming to the 3NF schema."""

from datetime import datetime
from sqlalchemy import Column, Integer, SmallInteger, String, Decimal, DateTime, ForeignKey, Boolean, Text, UniqueConstraint, TIMESTAMP
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from core.database.connection import Base

class AssetClass(Base):
    """SQLAlchemy model for the asset_classes table."""
    __tablename__ = "asset_classes"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    settlement_type = Column(String(20), nullable=True)
    default_locked_days = Column(Integer, default=0)

    tickers = relationship("Ticker", back_populates="asset_class")


class Ticker(Base):
    """SQLAlchemy model for the tickers table."""
    __tablename__ = "tickers"

    id = Column(SmallInteger, primary_key=True, autoincrement=True)
    asset_class_id = Column(Integer, ForeignKey("asset_classes.id", onupdate="CASCADE", ondelete="RESTRICT"), nullable=False)
    symbol = Column(String(20), unique=True, nullable=False)
    name = Column(String(255), nullable=True)
    exchange = Column(String(50), nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=func.now())

    asset_class = relationship("AssetClass", back_populates="tickers")
    ohlcv_records = relationship("OHLCV", back_populates="ticker", cascade="all, delete-orphan")


class TickerMetadata(Base):
    """SQLAlchemy model for the ticker_metadata table."""
    __tablename__ = "ticker_metadata"

    ticker_id = Column(SmallInteger, ForeignKey("tickers.id", onupdate="CASCADE", ondelete="CASCADE"), primary_key=True)
    settlement_days = Column(Integer, nullable=True)
    liquidity_type = Column(String(30), nullable=True)
    lot_size = Column(Integer, nullable=True)
    tick_size = Column(Decimal(12, 6), nullable=True)
    price_limit_up = Column(Decimal(12, 6), nullable=True)
    price_limit_down = Column(Decimal(12, 6), nullable=True)
    trading_fee = Column(Decimal(12, 6), nullable=True)
    allow_short = Column(Boolean, default=False)


class OHLCV(Base):
    """SQLAlchemy model for the ohlcv table."""
    __tablename__ = "ohlcv"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker_id = Column(SmallInteger, ForeignKey("tickers.id", onupdate="CASCADE", ondelete="CASCADE"), nullable=False)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Decimal(18, 6), nullable=True)
    high = Column(Decimal(18, 6), nullable=True)
    low = Column(Decimal(18, 6), nullable=True)
    close = Column(Decimal(18, 6), nullable=True)
    adjusted_close = Column(Decimal(18, 6), nullable=True)
    volume = Column(Integer, nullable=True)
    source = Column(String(100), nullable=True)
    data_quality = Column(String(30), nullable=True)
    created_at = Column(DateTime, default=func.now())

    ticker = relationship("Ticker", back_populates="ohlcv_records")

    __table_args__ = (
        UniqueConstraint("ticker_id", "timestamp", name="uq_ohlcv_ticker_time"),
    )
