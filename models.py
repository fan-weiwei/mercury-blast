from typing import List

class AnnotatedRecord:

    def __init__(self, name: str, annotations: List[str]) -> object:
        self.name = name
        self.annotations = annotations
