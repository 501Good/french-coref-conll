from dataclasses import dataclass


@dataclass
class CoNLLTokenLocation:
    sent_id: int
    token_id: int

    def __hash__(self):
        return hash((self.sent_id, self.token_id))

    def __repr__(self):
        return f"{self.sent_id}#{self.token_id}"


class WordIndex:
    def __init__(self, section: int = 0, utterance: int = 0, word: int = 0):
        self._section = section
        self._utterance = utterance
        self._word = word

    @classmethod
    def from_string(cls, string: str) -> None:
        """Initialise a word index from a string like "#s19.u20.w9"."""
        try:
            if string.endswith(".dash"):
                s, u, w, _ = string.split(".")
            else:
                s, u, w = string.split(".")
        except ValueError:
            raise ValueError(f'The string must be in format "#s0.u0.w0" for got "{string}" instead!')
        if s.startswith("#"):
            s = s[1:]
        return cls(int(s[1:]), int(u[1:]), int(w[1:]))

    @property
    def s(self):
        return self._section

    @s.setter
    def s(self, value):
        self._section = int(value)

    @property
    def u(self):
        return self._utterance

    @u.setter
    def u(self, value):
        self._utterance = int(value)

    @property
    def w(self):
        return self._word

    @w.setter
    def w(self, value):
        self._word = int(value)

    def __repr__(self):
        return f"s{self.s}.u{self.u}.w{self.w}"

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        if isinstance(other, WordIndex):
            return self.s == other.s and self.u == other.u and self.w == other.w
        return False
