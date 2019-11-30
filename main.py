from flow import *
from sys import argv 

def main():
  assert len(argv[1]) > 0
  model = train(epoch_count=2, batch_size=8, hr_images=HR_IMAGES[:100], lr_images=LR_IMAGES[:100])
  model.save_weights(argv[1])
	
  
if __name__== "__main__":
  main()
