import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
  ''' 
  Function to show image '''
  img = img / 2 + 0.5 # unnormalize
  npimg = img.numpy() # convert to numpy objects
  plt.imshow(np.transpose(npimg, (1, 2, 0)))
  plt.show()

def show_tensorboard_losses(writer,losses, total_iterations, mode = "train"):
    """
    Add training losses to the tensorboard writer
    Arguments:
      losses: losses to be added to the tensorboard summary writer, takes float values
      writer: Tensorboard summary writer
      total_iterations: No of iterations passed, takes int values
    """
    writer.add_scalars('Loss', {
               mode: losses,
                },total_iterations)       
        
