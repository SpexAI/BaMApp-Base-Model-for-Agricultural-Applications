import torch
import torchvision.transforms as T
import deeplake
import os
import glob
from PIL import Image
import numpy as np
import multiprocessing
import sys


class Upload:
    def __init__(self):
        self.url = 'hub://bamapp/test_large_small_embeddings'
        try:
            self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.model_small = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f'Error loading model: {e}')
            sys.exit(1)

        # check if cuda is available
        if torch.cuda.is_available():
            self.model.to('cuda')
            self.model_small.to('cuda')

    @staticmethod
    def get_image_files(directory) -> list[str]:
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
        """
        Extract the features from the image using the dino pretrained model.
        Features are computed on a resized image, using 224x224 pixels, and normalized to the imagenet mean/std.
        :param image: PIL image
        :return: features
        """

        # Preprocess image for the model
        preprocess = T.Compose([
            T.Resize(size=(518,518)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # Imagenet standards
        ])

        image = preprocess(image)[:3].unsqueeze(0)

        # Check if cuda is available
        if torch.cuda.is_available():
            image = image.to('cuda')

        with torch.no_grad():
            features_dict = self.model.forward_features(image)
            features_l = features_dict['x_norm_patchtokens']

            features_dict = self.model_small.forward_features(image)
            features_s = features_dict['x_norm_patchtokens']

        return features_l.cpu().numpy(), features_s.cpu().numpy()

    def upload(self, folder, commit_message):
        """
        Upload images from a folder to deeplake
        :param folder: Folder with images, will be searched recursively, should contain jpg, jpeg or png files
        :param commit_message: Commit message to use for the upload
        """
        image_files = self.get_image_files(folder)
        assert len(image_files) > 0, f'No images found in {folder}'
        try:
            # Load the dataset it should already exist, but if not use the code below to create it first.
            ds = deeplake.load(self.url)
        except Exception as e:
            print(f'Creating dataset as it does not seem to exist yet. {e}')
            ds = deeplake.empty(self.url)
            with ds:
                ds.create_tensor('images', htype='image', sample_compression='jpeg')
                ds.create_tensor('embeddings', htype='embedding')
                ds.create_tensor('embeddings_large', htype='embedding')

        @deeplake.compute
        def images_2_deeplake(image_file, sample_out):
            try:
                image = Image.open(image_file).convert('RGB')
                preprocess = T.Compose([
                    T.Resize(1024),
                    # T.CenterCrop(224),
                ])
                scaled_image = preprocess(image)
                image_embedding_large, image_embedding_small = self.extract_features(image)
                sample_out.append({'images': np.array(scaled_image), 'embeddings': image_embedding_small,
                                   'embeddings_large': image_embedding_large})
            except Exception as e:
                print(f'Failed to process {image_file}: {e}')

        # Use all but 2 cores, we still want to be able to control the computer
        num_workers = min(multiprocessing.cpu_count()-2, 1) if multiprocessing.cpu_count() > 2 else 1
        # create a checkpoint every 200 images
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
    assert args.folder is not None, 'Please specify a folder with images'
    assert args.commit_message is not None, 'Please specify a commit message'
    # check if args.folder exists
    assert os.path.isdir(args.folder), f'Folder {args.folder} is not a directory'
    uploader = Upload()
    uploader.upload(args.folder, args.commit_message)


def test():
    uploader = Upload()
    uploader.upload('/media/ben/DataTwo/Tmp_Foundation_Test', 'test')


if __name__ == '__main__':
    main()

