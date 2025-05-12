import os, random, shutil

base_dir = os.getcwd()
target_dir = os.path.join(base_dir, 'sample_data_from_cf_all')
os.makedirs(target_dir, exist_ok=True)

for i in range(1, 78):
    subfolder = os.path.join(base_dir, str(i))
    if os.path.isdir(subfolder):
        csv_files = [f for f in os.listdir(subfolder) if f.endswith('.csv')]
        if len(csv_files) >= 2:
            for file_name in random.sample(csv_files, 2):
                shutil.move(os.path.join(subfolder, file_name), os.path.join(target_dir, file_name))
