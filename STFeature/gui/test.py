from network import C3D_model
import torch
print("load")
model = C3D_model.C3D(num_classes=4, pretrained=False)
print("1111")
checkpoint = torch.load("model/C3D-uex_class4_epoch-79.pth.tar")
# torch.save(checkpoint, "model/C3D-uex_class4.pth", _use_new_zipfile_serialization=False)
# checkpoint = torch.load(self.model_pth_dir, map_location=lambda storage, loc: storage)
print("model done")

