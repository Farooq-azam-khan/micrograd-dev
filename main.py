import math


class Value:
    def __init__(self, data: int, _children=(), _op="", label=""):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

        self._backward = lambda: None
        self.grad = 0.0

    def backward(self):
        topo = []

        def build_topo(v):
            visited = set()
            if v not in visited:
                visited.add(v)
                for child in list(v._prev):
                    build_topo(child)
                topo.append(v)
            return topo

        topo = build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __repr__(self):
        return f"Value({self.data:.4E})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")

        out = Value(
            self.data + other.data,
            _children=(self, other),
            _op="+",
            label=f"(+ {self.label} {other.label})",
        )

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):  # other * self
        return self * other

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, label=f"{other}")
        out = Value(
            self.data * other.data,
            _children=(self, other),
            _op="*",
            label=f"(* {self.label} {other.label})",
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rtruediv__(self, other):  # other / self = (self**-1) * other
        return self**-1 * other

    def __truediv__(self, other):  # self / other = self * other**-1
        return self * other**-1

    def __pow__(self, other):  # self**other
        assert isinstance(other, (int, float)), "only supports float and powers"
        out = Value(
            self.data**other, (self,), _op=f"**{other}", label=f"{self.label}**{other}"
        )

        def _backward():
            self.grad += (other * self.data ** (other - 1)) * out.grad

        self._backward = _backward

        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self = -self + other
        return -self + other

    def exp(self):
        out = Value(
            data=math.exp(self.data),
            label=f"exp({self.label})",
            _children=(self,),
            _op="exp",
        )

        def _backward():
            self.grad += out.grad * out.data

        out._backward = _backward
        return out

    def tanh(self):
        # https://en.wikipedia.org/wiki/Hyperbolic_functions
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(data=t, _children=(self,), _op="tanh", label=f"tanh({self.label})")

        def _backward():
            self.grad += out.grad * (1 - t**2)

        out._backward = _backward

        return out
