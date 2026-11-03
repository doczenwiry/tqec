"""Defines the color representation used in the TQEC interop layer."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class RGBA:
    """RGBA representation of a color.

    Attributes:
        r: Red component of the color, in the range [0, 255].
        g: Green component of the color, in the range [0, 255].
        b: Blue component of the color, in the range [0, 255].
        a: Alpha component of the color, in the range [0, 1].

    """

    r: int
    g: int
    b: int
    a: float

    def with_alpha(self, a: float) -> RGBA:
        """Return a new RGBA with the same color but a different alpha value.

        Args:
            a: The new alpha value.

        Returns:
            A new RGBA with the same color but a different alpha value.

        """
        return RGBA(self.r, self.g, self.b, a)  # pragma: no cover

    def as_floats(self) -> tuple[float, float, float, float]:
        """Return the color as a tuple of floats. RGB values are normalized to the range [0, 1].

        Returns:
            Length 4 tuple of floats representing the color.

        """
        return (self.r / 255, self.g / 255, self.b / 255, self.a)

    def to_hex(self) -> str:
        """Return the color as a hexadecimal string.

        Returns:
            The color as a hexadecimal string.

        """
        return f"#{self.r:02x}{self.g:02x}{self.b:02x}"  # pragma: no cover


class TQECColor(Enum):
    """Predefined and commonly used colors in TQEC."""

    X = "X"
    Y = "Y"
    Z = "Z"
    H = "H"
    T = "T"
    CULTIVATION = "CULTIVATION"
    X_CORRELATION = "X_CORRELATION"
    Z_CORRELATION = "Z_CORRELATION"

    @property
    def rgba(self) -> RGBA:
        """Return the RGBA representation of the color."""
        match self:
            case TQECColor.X:
                color = RGBA(255, 127, 127, 1.0)
            case TQECColor.Y:
                color = RGBA(99, 198, 118, 1.0)
            case TQECColor.Z:
                color = RGBA(115, 150, 255, 1.0)
            case TQECColor.H:
                color = RGBA(255, 255, 101, 1.0)
            case TQECColor.T:
                color = RGBA(217, 102, 217, 1.0)
            case TQECColor.CULTIVATION:
                color = RGBA(65, 65, 65, 1.0)
            case TQECColor.X_CORRELATION:
                color = RGBA(255, 0, 0, 0.8)
            case TQECColor.Z_CORRELATION:
                color = RGBA(0, 0, 255, 0.8)

        return color

    def with_zx_flipped(self) -> TQECColor:
        """Return a ``X`` or ``Z`` color from a ``Z`` or ``X`` color and vice versa."""
        if self == TQECColor.X:
            return TQECColor.Z
        if self == TQECColor.Z:
            return TQECColor.X
        if self == TQECColor.X_CORRELATION:
            return TQECColor.Z_CORRELATION
        if self == TQECColor.Z_CORRELATION:
            return TQECColor.X_CORRELATION
        return self
