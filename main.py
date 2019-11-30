from flow import *
from sys import argv 

def main():
  assert len(argv[1]) > 0
  model = mkmodel()
  try:
    train(
      gan=model,
      epoch_count=100,
      batch_size=64,
      hr_images=HR_IMAGES[:50000],
      lr_images=LR_IMAGES[:50000]
    )
  except KeyboardInterrupt:
    pass
  model.save_weights(argv[1])
	
  
if __name__== "__main__":
  main()
