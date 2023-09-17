import os
import SimpleITK as sitk
import numpy as np


def get_xy_min_of_one(label_array):
    indices_of_one = np.nonzero(label_array == 1)

    if len(indices_of_one[0]) == 0:
        return None

    x_indices = indices_of_one[2]
    y_indices = indices_of_one[1]

    x_min = np.min(x_indices)
    y_min = np.min(y_indices)

    return x_min, y_min


if __name__ == '__main__':

    folder_path = "/infers" # The folder where the predicted results of the nnUNet att model are saved
    output_folder_path = "/infers_corr" # Folder saved after location correction

    # If the folder does not exist, create an output folder
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # The range of length and width obtained through statistical analysis of the dataset
    max_x = 320
    max_y = 256

    for filename in os.listdir(folder_path):
        if filename.endswith(".nii.gz"):
            file_path = os.path.join(folder_path, filename)

            #Read predicted label file
            label_image = sitk.ReadImage(file_path)
            label_array = sitk.GetArrayFromImage(label_image)

            xy_min = get_xy_min_of_one(label_array)

            if xy_min:
                x_min, y_min = xy_min
                x_max = x_min + max_x
                y_max = y_min + max_y
                if x_max > 512:
                    x_max = 512
                if y_max > 512:
                    y_max = 512

                #Create an all 0 array
                modified_array = np.zeros(label_array.shape, dtype=label_array.dtype)

                # Keep the part of the rectangular area with the original label 1
                modified_array[:, y_min:y_max, x_min:x_max] = label_array[:, y_min:y_max, x_min:x_max]

                # Create and save new SimpleITK image
                modified_image = sitk.GetImageFromArray(modified_array)
                modified_image.CopyInformation(label_image)

                output_file_path = os.path.join(output_folder_path, filename)
                sitk.WriteImage(modified_image, output_file_path)

                print(f"Modified file {filename} saved.")
            else:
                print(f"For file {filename}, no pixels with label 1.")