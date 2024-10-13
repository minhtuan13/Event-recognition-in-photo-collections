import os
import requests

arch = 'resnet50'

# Load the pre-trained weights
model_file = f'{arch}_places365.pth.tar'

# Check if the file exists and is writable
if not os.path.isfile(model_file) or not os.access(model_file, os.W_OK):
    weight_url = f'https://drive.google.com/file/d/1--E--vupQHiiPfOh8_hSRrjtewmG5G7G/view?usp=drive_link'
    print(f'Downloading weights from {weight_url}')
    
    # Make a request to download the file
    response = requests.get(weight_url, stream=True)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file with write-binary mode
        with open(model_file, 'wb') as f:
            # Write the content to the file in chunks
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print('Download complete.')
    else:
        print('Failed to download the weights. Status code:', response.status_code)
else:
    print('Model file is already present and accessible.')
