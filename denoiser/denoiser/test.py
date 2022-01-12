from .demucs import Demucs
from torchinfo import summary as summary_

model = Demucs()
model.to('cuda')
summary_(model,input_size=(32 ,4, 64000))

del model