import torch
import torchvision.transforms as T
import deeplake
import os
import glob
from PIL import Image
import numpy as np
import multiprocessing



class Upload:
    def __init__(self):
        self.url = 'hub://bamapp/test'
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')


    def get_image_files(self, directory):
        """
        Get image files from a directory
        :param directory:
        :return: list of image files
        """
        image_files = []
        for extension in ['jpg', 'jpeg', 'png']:
            image_files.extend(glob.glob(f'{directory}/**/*.{extension}', recursive=True))
        return [os.path.abspath(image) for image in image_files]



    def extract_features(self, image):
        '''
        Extract the features from the image using the dino pretrained model.
        Features are computed on a resized image, using 224x224 pixels, and normalized to the imagenet mean/std.
        :param image: PIL image
        :return: features
        '''

        # Preprocess image for the model
        preprocess = T.Compose([
            T.Resize(size=(518,518)),
            #T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # Imagenet standards
        ])
        image = preprocess(image)[:3].unsqueeze(0)

        with torch.no_grad():
            features_dict = self.model.forward_features(image)
            features = features_dict['x_norm_patchtokens']

        return features.cpu().numpy()

    def upload(self, folder, commit_message):
        image_files = self.get_image_files(folder)
        ds = deeplake.load(self.url) # Load the dataset it should already exist, but if not use the code below to create it first.
        #ds = deeplake.empty(self.url, overwrite=True)
        #with ds:
        #    ds.create_tensor('images', htype='image', sample_compression='jpeg')
        #    ds.create_tensor('embeddings', htype='embedding')

        @deeplake.compute
        def images_2_deeplake(image_file, sample_out):

            image = Image.open(image_file).convert('RGB')
            preprocess = T.Compose([
                T.Resize(518),
                # T.CenterCrop(224),
            ])
            scaled_image = preprocess(image)
            image_embedding = self.extract_features(image)

            sample_out.append({'images': np.array(scaled_image), 'embeddings': image_embedding})
        num_workers = min(multiprocessing.cpu_count()-2, 1) # Use all but 2 cores, we still want to be able to control the computer
        checkpoint_interval = min(200, len(image_files))
        images_2_deeplake().eval(image_files, ds, num_workers=num_workers, checkpoint_interval=checkpoint_interval)
        ds.commit(commit_message)
        print(f'Uploaded {len(image_files)} images to {self.url}')
        print(f'Logs: {ds.log()}')


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Upload images to deeplake')
    parser.add_argument('--folder', type=str, help='Folder with images')
    parser.add_argument('--commit_message', type=str, help='Commit message')
    args = parser.parse_args()
    uploader = Upload()
    uploader.upload(args.folder, args.commit_message)

def test():
    uploader = Upload()
    uploader.upload('/media/ben/DataTwo/Tmp_Foundation_Test', 'test')


if __name__ == '__main__':
    main()