import torch

# import your model class
from Model_3_Final.member3_train_modelUpgrade import YModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# rebuild model architecture
model = YModel(freeze_backbone=False).to(device)

# load weights
model.load_state_dict(torch.load("best_ymodel.pth", map_location=device))

model.eval()

# save full model
torch.save(model, "best_ymodel_FULL.pth")

print("✅ Full model saved successfully!")