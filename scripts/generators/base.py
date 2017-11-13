from abc import ABCMeta, abstractmethod

class BaseGenerator(metaclass=ABCMeta):
    @abstractmethod
    def generator(self, index_list=None):
        if index_list is None:
            raise Exception("Invalid argument : index_list is None.")
        pass

    @abstractmethod
    def input_shape(self):
        pass