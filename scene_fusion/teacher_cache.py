"""Bound GPU source-model memory independently of the number of captures."""
from collections import OrderedDict

class TeacherCache:
    def __init__(self,folder,count,capacity=2,loader=None):
        if capacity<1:raise ValueError('Positive cache capacity required')
        if loader is None:
            from refit_render import load_scene
            loader=load_scene
        self.folder,self.count,self.capacity,self.loader=folder,count,capacity,loader
        self.models=OrderedDict()
    def __getitem__(self,i):
        if not 0<=i<self.count:raise IndexError(i)
        if i not in self.models:
            while len(self.models)>=self.capacity:self.models.popitem(last=False)
            self.models[i]=self.loader(self.folder/f'teacher-{i}.npy')
        self.models.move_to_end(i)
        return self.models[i]
