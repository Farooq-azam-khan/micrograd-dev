class Value:
    def __init__(self, data: int, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value({self.data:.4E})"

    def __add__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
            label=f"(+ {self.label} {other.label})",
        )

    def __mul__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
            label=f"(* {self.label} {other.label})",
        )

    def __truediv__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.data / other.data,
            _children=(self, other),
            _op="/",
            label=f"(/ {self.label} {other.label})",
        )

    def __sub__(self, other):
        if not isinstance(other, Value):
            other = Value(other)
        return Value(
            self.data - other.data,
            _children=(self, other),
            _op="-",
            label=f"(- {self.label} {other.label})",
        )
