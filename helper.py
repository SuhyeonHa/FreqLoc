import os
import torch
from PIL import Image
import torchvision.transforms as transforms
import torchvision
import torch.nn as nn
import torch.nn.functional as F

# Load images 
class Denormalize(transforms.Normalize):
    def __init__(self, mean, std):
        mean = torch.tensor(mean)
        std = torch.tensor(std)
        self.mean_rev = -mean / std
        self.std_rev = 1 / std
        super().__init__(mean=self.mean_rev, std=self.std_rev)

norm_imagenet = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
denorm_imagenet = Denormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

def load_images_from_path(path: str, num_images: int = 1, transform=None) -> torch.Tensor:
    images = []
    file_names = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    
    for filename in sorted(os.listdir(path)):
        if filename.lower().endswith(valid_extensions):
            image_path = os.path.join(path, filename)
            try:
                image = Image.open(image_path).convert("RGB")
                images.append(transform(image))
                file_names.append(filename)
            except Exception as e:
                print(f"Warning: Could not load image {image_path}. Error: {e}")

    if not images:
        raise FileNotFoundError(f"No valid images found in the specified path: {path}")

    return torch.stack(images[2:3]), file_names[2:3]


def load_image(path: str, transform=None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

# Logging

class Tee:
    def __init__(self, stream1, stream2):
        self.stream1 = stream1
        self.stream2 = stream2

    def write(self, message):
        self.stream1.write(message)
        self.stream2.write(message)
        self.flush()

    def flush(self):
        self.stream1.flush()
        self.stream2.flush()

# Save images
def save_images(results, filename, save_dir):
    names = ["original", "watermarked", "prediction", "watermark"]

    for img, name in zip(results, names):
        output_dir = os.path.join(save_dir, name)
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"{filename}")
        if name == "watermark":
            img = img*10
        torchvision.utils.save_image(img, save_path)

    # Save all images at once
    combined = []
    for tensor in results:
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        combined.append(tensor)

    combined = torch.cat(combined, dim=0)
    os.makedirs(os.path.join(save_dir, "results"), exist_ok=True)
    save_path = os.path.join(save_dir, "results", f"{filename}")
    torchvision.utils.save_image(combined, save_path, nrow=3)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        # elif self.task_type == 'multi-class':
        #     return self.multi_class_focal_loss(inputs, targets)
        # elif self.task_type == 'multi-label':
        #     return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss