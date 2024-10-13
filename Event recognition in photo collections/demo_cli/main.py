# lib
from PIL import Image
import torch.utils.data as data
import itertools
import os
import numpy as np
import argparse

import torch
from PIL import Image
import torch.nn as nn
import datetime
import pickle
import joblib
from torchvision import models, transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize
import exifread

def default_loader(path):
    # List of common image extensions
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']

    # Check if the provided path exists
    if not os.path.exists(path):
        for ext in img_extensions:
            img_path = path + ext
            if os.path.exists(img_path):
                img = Image.open(img_path)
                return img.convert('RGB')
            # If no image file found with common extensions, raise an error
            # raise FileNotFoundError("No image file found with common extensions in the provided path.")
    else:
        # If the path has an extension, directly open the image
        img = Image.open(path)
        return img.convert('RGB')

class ObjectCNN (nn.Module):
    def __init__(self):
        super(ObjectCNN, self).__init__()
        model_object = models.alexnet(pretrained=True)
        model_object.classifier = torch.nn.Sequential(*list(model_object.classifier.children())[:-2])
        # Freeze the parameters of the feature extractor (convolutional layers)
        for param in model_object.features.parameters():
            param.requires_grad = False
        model_object.eval()
        self.alex =  model_object
    def forward(self, image):
        obj_extract = self.alex
        #image= default_loader(image_path)
        image_tensor = self.preprocess_object(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = obj_extract(image_tensor)
        return features.flatten()
    def preprocess_object(self, image):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)
class ObjectBlock (nn.Module):
    def __init__(self):
        super (ObjectBlock, self).__init__()
        self.layer = ObjectCNN()

    def forward (self, collection):
        object_extractor = self.layer
        obj_features = []
        for image in collection:
            # Extract features for each image
            image_obj_features = object_extractor(image)
            # Add the features to the lists
            obj_features.append(image_obj_features)
        # Calculate the average of object features
        avg_obj_features = sum(obj_features) / len(obj_features)
        return avg_obj_features
    
class SceneCNN (nn.Module):
    def __init__(self):
        from torchvision import models
        super(SceneCNN, self).__init__()
        arch = 'resnet50'

        # load the pre-trained weights
        model_file = '%s_places365.pth.tar' % arch
        if not os.access(model_file, os.W_OK):
            weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
            os.system('wget ' + weight_url)

        model_scene = models.__dict__[arch](num_classes=365)
        checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        model_scene.load_state_dict(state_dict)
        model_scene.eval()
        for param in model_scene.parameters():
            param.requires_grad = False
        self.resnet50 =  model_scene
    def forward(self, image):
        scn_extract = self.resnet50
        #image = default_loader(image_path)
        image_tensor = self.preprocess_scene(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = scn_extract(image_tensor)
        return features.flatten()
    def preprocess_scene(self, image):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return transform(image)

class SceneBlock (nn.Module):
    def __init__(self):
        super (SceneBlock, self).__init__()
        self.layer = SceneCNN()
    def forward (self, collection):
        scene_extractor = self.layer
        scn_features = []
        for image in collection:
            # Extract features for each image
            image_scn_features = scene_extractor(image)
            # Add the features to the lists
            scn_features.append(image_scn_features)
        # Calculate the average of object features
        avg_scn_features = sum(scn_features) / len(scn_features)

        return avg_scn_features
class TimeBlock(nn.Module):
    def __init__(self):
        super(TimeBlock, self).__init__()
        # self.time_features = self.extract_time_features
        # self.collection_feature = self.compute_collection_duration
    def extract_time_features(self, timestamps):
        # Convert Unix timestamps to datetime objects
        dt_objects = [datetime.datetime.fromtimestamp(ts) for ts in timestamps]

        # Extract time features
        years = [dt.year for dt in dt_objects]
        months = [dt.month for dt in dt_objects]
        days = [dt.day for dt in dt_objects]
        weekdays = [dt.weekday() for dt in dt_objects]  # 0 for Monday, 6 for Sunday

        # Calculate the average of each feature
        avg_year = torch.tensor(sum(years) / len(years), dtype=torch.long)
        avg_month = torch.tensor(sum(months) / len(months), dtype=torch.long)
        avg_day = torch.tensor(sum(days) / len(days), dtype=torch.long)
        avg_weekday = torch.tensor(sum(weekdays) / len(weekdays), dtype=torch.long)

        return avg_year, avg_month, avg_day, avg_weekday

    def compute_collection_duration(self, timestamps):
        # Compute the duration in days between the earliest and latest timestamps in the collection
        earliest = torch.min(timestamps)
        latest = torch.max(timestamps)
        earliest_dt = datetime.datetime.fromtimestamp(earliest.item())
        latest_dt = datetime.datetime.fromtimestamp(latest.item())
        duration = (latest_dt - earliest_dt).days
        return duration

    def extract_photo_time_features(self, photo_timestamp):
        # Extract time features for an individual photo
        return torch.stack(self.extract_time_features(photo_timestamp))
    def compute_photo_collection_time_features(self, timestamps):
        # Compute collection-level time features for a set of photo timestamps and timestamps
        collection_duration = self.compute_collection_duration(timestamps)
        return torch.tensor(collection_duration, dtype=torch.long)
    def forward(self, timestamps):
        # Forward pass to compute combined 5-dimensional feature vector
        timestamps_tensor = torch.tensor(timestamps)
        photo_time_features = self.extract_photo_time_features(timestamps_tensor)

        collection_time_features = self.compute_photo_collection_time_features(timestamps_tensor)
        combined_features = torch.cat([photo_time_features, collection_time_features.unsqueeze(0)])

        return combined_features

class HSM_ER (nn.Module):
    def __init__(self):
        super(HSM_ER, self).__init__()
        self.object_extractor = ObjectBlock()
        self.scene_extractor = SceneBlock()
        self.time_extractor = TimeBlock()
        self.svm_coarse = joblib.load('pretrained/coarse_model_2.joblib')
        self.fine_classifiers_scn = {}
        self.fine_classifiers_obj = {}
        self.fine_classifiers_time = {}
        self.groups = {
            0: [0, 1],
            1: [2, 3, 7],
            2: [4, 5, 6, 8],
            3: [9, 10, 11],
            4: [12, 13]
            }
        for i in range(5):
            self.fine_classifiers_scn[i] = joblib.load(f'pretrained/fine_model_scn_{i}_2.joblib')
            self.fine_classifiers_obj[i] = joblib.load(f'pretrained/fine_model_obj_{i}_2.joblib')
            self.fine_classifiers_time[i] = joblib.load(f'pretrained/fine_model_time_{i}_2.joblib')




    def forward(self, collection, collection_timestamp):
        scene_features = self.scene_extractor(collection)
        object_features = self.object_extractor(collection)
        time_features = self.time_extractor(collection_timestamp)




        ## transfer to numpy
        scene_features = scene_features.detach().numpy().reshape(1, -1)
        object_features = object_features.detach().numpy(). reshape(1, -1)
        time_features = time_features.detach().numpy().reshape(1, -1)
        ## normalize L2 applied for scene and objec feaure
        scene_features = normalize(scene_features, norm='l2')
        object_features = normalize(object_features, norm='l2')
        ## normalize MinMax applied for time feature
        time_features = MinMaxScaler().fit_transform(time_features)

        out = self.predict(scene_features, object_features, time_features)
        return out
    def get_predictions(self, X_sample_scn, X_sample_obj, X_sample_time):
        coarse_prediction_proba = self.svm_coarse.predict_proba(X_sample_scn)

        fine_predictions_proba_scn = []
        fine_predictions_proba_obj = []
        fine_predictions_proba_time = []
        #X_sample = np.concatenate((X_sample_scn, X_sample_obj, X_sample_time), axis=1)
        for coarse_label, fine_classifier in self.fine_classifiers_scn.items():
            fine_predictions_proba_scn.append(fine_classifier.predict_proba(X_sample_scn))
        for coarse_label, fine_classifier in self.fine_classifiers_obj.items():

            fine_predictions_proba_obj.append(fine_classifier.predict_proba(X_sample_obj))
        for coarse_label, fine_classifier in self.fine_classifiers_time.items():
            fine_predictions_proba_time.append(fine_classifier.predict_proba(X_sample_time))
        res1 = np.zeros((1, 14))
        res2 = np.zeros((1, 14))
        res3 = np.zeros((1, 14))

        # Fill the result array using the groups and arrays
        for group_index, indices in self.groups.items():
            for array_index, index in enumerate(indices):
                res1[0, index] = fine_predictions_proba_scn[group_index][0, array_index]
                res2[0, index] = fine_predictions_proba_obj[group_index][0, array_index]
                res3[0, index] = fine_predictions_proba_time[group_index][0, array_index]
        a = 0.6987260282926773
        b = 0.1367394511795748

        final_proba = a*coarse_prediction_proba + b*res2 + (1-a-b)*res3
        result = np.argmax(final_proba)
        return result
    def predict(self, X_sample_scn, X_sample_obj, X_sample_time):
        X_sample_scn = X_sample_scn.reshape(1, -1)
        X_sample_obj = X_sample_obj.reshape(1, -1)
        X_sample_time = X_sample_time.reshape(1, -1)

        # Make predictions for the current sample
        
        final_prediction = self.get_predictions(X_sample_scn, X_sample_obj, X_sample_time)


        return final_prediction
    

label_dict = {
    'cruise': 0, 'exhibition': 1, 'christmas': 2, 'halloween': 3, 'hiking': 4, 'road_trip': 5,
    'concert': 6, 'birthday': 7, 'skiing': 8, 'saint_patricks_day': 9, 'wedding': 10, 'graduation': 11,
    'easter': 12, 'children_birthday': 13
}

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def get_last_modified_date(file_path):
    # Lấy thời gian sửa đổi cuối cùng (timestamp)
    timestamp = os.path.getmtime(file_path)
    timestamp = int(timestamp)
    return timestamp

def get_capture_date(img_path):
    with open(img_path, 'rb') as f:
        tags = exifread.process_file(f)

    # Lấy giá trị ngày tháng từ thẻ EXIF
    date_time = str(tags.get('EXIF DateTimeOriginal'))
    if date_time != 'None':
        date_time_obj = datetime.datetime.strptime(date_time, '%Y:%m:%d %H:%M:%S')

        # Chuyển đổi đối tượng datetime thành Unix timestamp
        unix_timestamp = int(date_time_obj.timestamp())
        return unix_timestamp
    
    return get_last_modified_date(file_path = img_path)
   
def getData (collection_path):

    images = []
    timestamps = []

    for item in os.listdir(collection_path):
        item_path = os.path.join(collection_path, item)
        if is_image_file(item_path):
            img = Image.open(item_path)
            images.append(img)
            timestamps.append(get_capture_date(img_path = item_path))
    return images, timestamps 
    


def main():
    parser = argparse.ArgumentParser(description="A simple CLI example.")
    parser.add_argument('--collection', type=str, help='Collection path')

    
    args = parser.parse_args()


    collection_path = args.collection
    #collection_path = './data/01'
    images, timestamps = getData(collection_path)

    model = HSM_ER()
    model.eval()

    predict = model(images, timestamps)
    reverse_label_dict = {v: k for k, v in label_dict.items()}
    pre_event = reverse_label_dict.get(predict)
    print("Event: ",pre_event)


if __name__ == "__main__":
    main()