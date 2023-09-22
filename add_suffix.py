import os

# 指定文件夹路径
folder_path = './nnUNet_raw_data_base/nnUNet_raw_data/Task001_Teeth/imagesTs'

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.nii.gz'):
        # 构造新的文件名
        new_filename = filename.replace('.nii.gz', '_0000.nii.gz')

        # 构造文件的完整路径
        old_filepath = os.path.join(folder_path, filename)
        new_filepath = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_filepath, new_filepath)
