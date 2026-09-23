import torch
from backend.src.models.mobilevit_v2 import MobileViT_v2
from backend.src.utils.loss import RoadExtractionLoss
from backend.src.data.dataset import get_train_transforms

print("1. Initializing Model...")
model = MobileViT_v2(num_classes=1, width_mult=1.0)
print("2. Initializing Loss...")
loss_fn = RoadExtractionLoss(total_epochs=10)

print("3. Generating dummy data...")
x = torch.randn(2, 3, 256, 256)
y = torch.randint(0, 2, (2, 1, 256, 256)).float()

print("4. Forward pass...")
logits = model(x)
print(f"Logits shape: {logits.shape}")

print("5. Loss computation...")
loss, comps = loss_fn(logits, y, return_components=True)
print(f"Loss successful: {comps}")
print("ALL TESTS PASSED!")
