class Value:
    def __init__(self, data: int, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        return Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
            label=f"(+ {self.label} {other.label})",
        )

    def __mul__(self, other):
        return Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
            label=f"(* {self.label} {other.label})",
        )
