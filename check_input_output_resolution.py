
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model with the number of classes (e.g., 21 for the VOC dataset)
num_classes = 21
model = FCN8s(num_classes=num_classes).to(device)

# Print the model summary (optional, for detailed insight)
# print(model)

# Create a sample input tensor of size (batch_size, channels, height, width).
# For example, a single (batch_size=1) 224x224 RGB image (channels=3).
input_tensor = torch.randn(1, 3, 224, 224).to(device)

# Forward pass of the sample input through the model
output = model(input_tensor)

# Check the output size, it should be (1, num_classes, 224, 224) for this sample
print('Output size:', output.size())

# Ensure no gradients accumulated in weights
model.zero_grad()