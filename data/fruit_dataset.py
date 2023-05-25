import torchvision 
import torchvision.transforms as transforms 
from torchvision.datasets import ImageFolder
import torch.utils.data

def get_loaders(config):
    
    """Get train, val and test loaders
    Arguments:
          config: config directory from train function
    Returns:
          train_loader, val_loader, test_loader
    """
    transform = {
        'train': transforms.Compose([
            transforms.Resize([224,224]), # Resizing the image as the VGG only take 224 x 244 as input size
            transforms.RandomHorizontalFlip(), # Flip the data horizontally
            transforms.RandomVerticalFlip(), # Flip the data horizontally
#             transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.05),
#             transforms.AugMix(),
#             transforms.RandomPerspective(distortion_scale=0.6, p=1.0),
#             transforms.RandomRotation(degrees=(0, 180)),
            #TODO if it is needed, add the random crop
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
        ])

    }
    
    dataset = ImageFolder(config['dataset_folder'], transform = transform['train'])
    dataset_len = len(dataset)
    train_split = int(dataset_len * 0.6)
    val_split = int(dataset_len * 0.2)
    test_split = dataset_len - train_split - val_split
    train_data, val_data, test_data = torch.utils.data.random_split(dataset, [train_split, val_split, test_split])
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True,    num_workers=config['num_workers'])
    val_loader = torch.utils.data.DataLoader(dataset=val_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    
    return train_loader, val_loader, test_loader