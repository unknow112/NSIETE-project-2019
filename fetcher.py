

# importing the requests library 
import requests
import os
import json
  
# api-endpoint 
URL = "https://pixabay.com/api/"
SAVE_DIR='/Users/vikival/Downloads/dirty'
  
# location given here
key='7487917-553ab463a47fc10f7503440b1'
q = "autumn"
image_type='photo'
  
# defining a params dict for the parameters to be sent to the API 
PARAMS = {'key': key,'q':q, 'image_type': image_type} 
  
# sending get request and saving the response as response object 
r = requests.get(url = URL, params = PARAMS) 
  
# extracting data in json format 
data = r.json() 

def dataMapper(image):
    return { 'url': image['largeImageURL'], 'tags' :image['tags'] }

imageIterator = list(map(dataMapper, data['hits']))

with open(os.path.join(SAVE_DIR, q+'.json'), 'w') as file:
    json.dump(imageIterator, file)

def fetchImage(image):
    response = requests.get(image['url'])
    return { **image, 'image': response.content }
    
imageIteratorWithImage = map(fetchImage, imageIterator)

for index, image in enumerate(imageIteratorWithImage):
    imageName = image['url'].split('/')[-1]
    print(index,': writing image with imageName', imageName)
    with open(os.path.join(SAVE_DIR, imageName), 'wb') as file:
        file.write(image['image'])
    
