from flow import *
from sys import argv 

def main():
  assert len(argv[1]) > 0
  model = train(epoch_count=10, batch_size=20, hr_images=HR_IMAGES, lr_images=LR_IMAGES)
  model.save_weights(argv[1])
	
  
if __name__== "__main__":
  main()
