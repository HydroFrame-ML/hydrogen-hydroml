import xbatcher as xb
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

@functional_datapipe("xbatcher")
class XbatcherDataPipe(IterDataPipe):
    def __init__(self, parent_pipe, input_dims, **kwargs):
        self.parent_pipe = parent_pipe
        self.input_dims = input_dims
        self.kwargs = kwargs

    def __iter__(self):
        for dataarray in self.parent_pipe:
            bgen = xb.BatchGenerator(dataarray, self.input_dims, **self.kwargs)
            for batch in bgen:
                yield batch

    def __len__(self):
        bgens = [xb.BatchGenerator(ds, self.input_dims, **self.kwargs) for ds in self.parent_pipe]
        return sum(len(bgen) for bgen in bgens)
