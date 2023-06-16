import torch
import torchvision.transforms as T
import deeplake
import os
import glob
from PIL import Image
import numpy as np
import multiprocessing
import sys
import datetime

class Upload:
    def __init__(self):
        self.url = 'hub://bamapp/bamapp'
        try:
            self.model_small = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        except Exception as e:
            print(f'Error loading model: {e}')
            sys.exit(1)

        # check if cuda is available
        if torch.cuda.is_available():
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

    @staticmethod
    def process_image(image):

        # Define transformations
        resize = T.Resize(1024, Image.BICUBIC)
        center_crop = T.CenterCrop(1024)

        # Find the longest edge of the image
        longest_edge = max(image.size)

        # Output list to hold the processed images
        output = []

        if longest_edge > 1024:
            # Resize the image
            resized_image = resize(image)
            output.append(np.array(resized_image))

            # Perform a center crop
            cropped_image = center_crop(resized_image)
            output.append(np.array(cropped_image))

            # Tile the image into 1024x1024 tiles if possible
            width, height = image.size
            if width > 1024 and height > 1024:
                for i in range(0, width - 1023, 1024):  # adjust the range to avoid going outside the image
                    for j in range(0, height - 1023, 1024):  # adjust the range to avoid going outside the image
                        box = (i, j, min(i + 1024, width),
                               min(j + 1024, height))  # adjust the box to avoid going outside the image
                        try:
                            tile = image.crop(box)
                            output.append(np.array(tile))
                        except:
                            pass
        else:
            # Just resize the image if it's smaller than 1024
            resized_image = resize(image)
            output.append(np.array(resized_image))

        return output

    def extract_features(self, image):
        """
        Extract the features from the image using the dino pretrained model.
        Features are computed on a resized image, using 224x224 pixels, and normalized to the imagenet mean/std.
        :param image: PIL image
        :return: features
        """

        # Preprocess image for the model
        preprocess = T.Compose([
            T.Resize(size=(224,224)), #518
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]), # Imagenet standards
        ])

        image = preprocess(image)[:3].unsqueeze(0)

        # Check if cuda is available
        if torch.cuda.is_available():
            image = image.to('cuda')

        with torch.no_grad():

            features_dict = self.model_small.forward_features(image)
            features_s = features_dict['x_norm_patchtokens']
            features_s = np.squeeze(features_s.cpu().numpy())
        return features_s.flatten()

    def upload(self, folder, commit_message, metadata={}):
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
                ds.create_tensor('images', htype='image', sample_compression='jpeg', create_sample_info_tensor=True)
                ds.create_tensor('embeddings', htype='embedding')
                ds.create_tensor('metadata', htype='json')

        @deeplake.compute
        def images_2_deeplake(image_file, sample_out):
            try:
                image = Image.open(image_file).convert('RGB')

                scaled_images = self.process_image(image)
                image_embedding_small = self.extract_features(image)
                for scaled_image in scaled_images:
                    sample_out.append({'images': np.array(scaled_image), 'embeddings': image_embedding_small,
                                       'metadata': metadata})

            except Exception as e:
                print(f'Failed to process {image_file}: {e}')

        # Use all but 2 cores, we still want to be able to control the computer
        num_workers = min(multiprocessing.cpu_count()-2, 1) if multiprocessing.cpu_count() > 2 else 1
        num_workers = min(num_workers, len(image_files))
        # create a checkpoint every 200 images
        checkpoint_interval = min(1000, len(image_files))
        images_2_deeplake().eval(image_files, ds, num_workers=num_workers, checkpoint_interval=checkpoint_interval)
        commit_response = ds.commit(commit_message)

        print(f'Commit message: {commit_response}, please keep this message for future reference')
        print(f'Uploaded {len(image_files)} images to {self.url}')
        print(f'Logs: {ds.log()}')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f'BaMApp_upload_log_{timestamp}.txt'
        try:
            # Open the log file and write commit message to it as well as the log
            with open(filename, 'w') as f:
                f.write(f'Commit message: {commit_message}\n')
                f.write(f'Metadata: {metadata}\n')
                f.write(f'Commit Id: {commit_response}\n')
                original_stdout = sys.stdout
                sys.stdout = f
                ds.log()
                sys.stdout = original_stdout
            print(f'Log file written to {filename}')
        except Exception as e:
            print(f'Failed to write log file: {e}')


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description='Upload images to deeplake')
    parser.add_argument('--folder', type=str, help='Folder with images')
    parser.add_argument('--commit_message', type=str, help='Commit message')
    parser.add_argument('--json', type=str, help='Json file formated Metadata eg. {"Origin": "Test"}')
    args = parser.parse_args()
    assert args.folder is not None, 'Please specify a folder with images'
    assert args.commit_message is not None, 'Please specify a commit message'
    metadata = {}
    assert args.json is not None, 'Please specify a json file with metadata and source_dataset_name'
    if args.json is not None:
        try:
            assert 'source_dataset_name' in args.json, 'Please specify a source_dataset_name in the json file'
            metadata = json.loads(args.json)
        except json.JSONDecodeError as e:
            json_string = args.json
            error_message = e.msg
            error_pos = e.pos
            error_line = e.lineno
            error_col = e.colno
            error_char = json_string[error_pos]

            error_context = json_string[max(0, error_pos - 3):error_pos + 4]

            print(f"JSON decoding error: {error_message}")
            print(f"Error occurred at line {error_line}, column {error_col}")
            print(f"Character '{error_char}' caused the error")
            print(f"Context: {error_context}")
            sys.exit(1)
    # check if args.folder exists
    assert os.path.isdir(args.folder), f'Folder {args.folder} is not a directory'
    uploader = Upload()
    uploader.upload(args.folder, args.commit_message, metadata=metadata)


if __name__ == '__main__':
    main()

