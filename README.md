# ProLesA-Net
The current repository serves as the placeholder for the ProLesA-Net model

# Model Architecture
ProLesA-Net along with the multiscale attention mechanisms are presented below:
![ProLesA-Net](ModelMaterials/ProlesaNet.png)
![MultiScale attention mechanisms](ModelMaterials/components.png)

# Usage
```python
import tensorflow as tf
import ProlesaModule

msqa = ProlesaModule.ProLesA_Net.ProlesaNet()
msqa.build(input_shape = [1,24, 192,192,3])

msqa.summary()
```
