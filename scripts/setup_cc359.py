import zipfile
import os
import glob
import shutil
from pathlib import Path
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('path', type=str)

args = parser.parse_args()

path = Path(args.path)
dl_path = path / 'download'

vendors = ['philips_3', 'philips_15', 'siemens_3', 'siemens_15', 'ge_15', 'ge_3']


# ----- Extract the gdrive zip file -----
print(f"Extracting downloaded zip file {path / 'CC359.zip'}")

with zipfile.ZipFile(path / 'CC359.zip', 'r') as zip_file:
    zip_file.extractall(path)

# rename
shutil.move(path / 'CC359', path / dl_path)
# os.remove(path / 'CC359.zip')


# ----- Extract Original.zip and re-organize -----
print(f"Extracting {dl_path / 'Original.zip'} and re-organize")

with zipfile.ZipFile(dl_path / 'Original.zip', 'r') as zip_file:
    zip_file.extractall(path)

# os.remove(dl_path / 'Original.zip')

for vendor in vendors:
    vendor_path = path / 'Original' / vendor.upper()
    vendor_path.mkdir(parents=True)

    for file_path in glob.iglob(str(path / 'Original' / f'CC*_{vendor}_*.nii.gz')):
        shutil.move(file_path, vendor_path)


# ----- Extract Silver-standard-STAPLE.zip and re-organize -----
silver_standard = "Silver-standard-machine-learning"
print(f"Extracting {dl_path / 'Skull-stripping-masks' / f'{silver_standard}.zip'} and re-organize")

with zipfile.ZipFile(dl_path / 'Skull-stripping-masks' / f'{silver_standard}.zip', 'r') as zip_file:
    zip_file.extractall(path)

# rename
# os.remove(dl_path / 'Skull-stripping-masks' / f'{silver_standard}.zip')

for vendor in vendors:
    vendor_path = path / 'Silver-standard' / vendor.upper()
    vendor_path.mkdir(parents=True)

    for file_path in glob.iglob(str(path / 'Silver-standard' / f'CC*_{vendor}_*.nii.gz')):
        shutil.move(file_path, vendor_path)

# final cleanup
shutil.rmtree(path / '__MACOSX')

print("Done.")
