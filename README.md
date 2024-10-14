# Event Recognition in Photo Collections

This project implements a hierarchical event recognition system based on a collection of photos. 
It extracts object, scene, and temporal features from images to classify events like weddings, hiking, birthdays, and more.

#Features
- Object Feature Extraction: Uses a pre-trained AlexNet model to extract object features from the images.
- Scene Feature Extraction: Uses a pre-trained ResNet-50 model trained on Places365 to extract scene features.
- Temporal Feature Extraction: Extracts temporal features from image timestamps (year, month, day, weekday, and duration).
- Coarse and Fine Classification: Uses pre-trained SVM classifiers to first predict coarse event categories and then refine them to specific event types based on object, scene, and time features.

#Dataset
The project works with collections of images in a folder. It reads images and extracts EXIF metadata to derive timestamps. If EXIF data is unavailable, the file's last modified date is used.

