from abc import ABC, abstractmethod


class BaseEditorEvaluation(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def evaluate_single_edit(self):
        pass
    
    @abstractmethod
    def evaluate_sequential_edit(self, edit_n = 10, random = False, seed = None):
        pass
