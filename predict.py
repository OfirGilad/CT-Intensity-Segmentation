import os
import csv
from utils import *

ct_filepath = "./Predict/PA000016.nii.gz"
output_path = "./Predict/output/"


def segment_lungs():
    myFile = open('Predict/output/lung_volumes.csv', 'w')
    lung_areas = []

    img_name = os.path.split(ct_filepath)[-1].split('.nii')[0]
    out_mask_name = output_path + img_name + "_mask"
    contour_name = output_path + img_name + "_contour.png"

    ct_img = nib.load(ct_filepath)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()
    ct_numpy = ct_numpy[:, :, 256]
    # print(ct_numpy.min(), ct_numpy.max())

    contours = intensity_seg(ct_numpy, min=-1000, max=-300)

    lungs = find_lungs(contours)
    show_contour(ct_numpy, lungs, contour_name, save=True)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs)
    save_nifty(lung_mask, out_mask_name, ct_img.affine)

    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))
    lung_areas.append([img_name, lung_area])  # int is ok since the units are already mm^2

    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(lung_areas)


def segment_vessels():
    # basepath = './Images/slice*.nii.gz'
    # vessels = './Vessels/'
    # overlay_path = './Vessel_overlayed/'
    # paths = sorted(glob.glob(basepath))
    myFile = open('Predict/output/vessel_volumes.csv', 'w')
    lung_areas_csv = []
    ratios = []

    def split_array_coords(array, indx=0, indy=1):
        x = [array[i][indx] for i in range(len(array))]
        y = [array[i][indy] for i in range(len(array))]
        return x, y

    def create_vessel_mask(lung_mask, ct_numpy, denoise=False):
        vessels = lung_mask * ct_numpy  # isolate lung area
        vessels[vessels == 0] = -1000
        vessels[vessels >= -500] = 1
        vessels[vessels < -500] = 0
        show_slice(vessels)
        if denoise:
            return denoise_vessels(lungs_contour, vessels)
        show_slice(vessels)

        return vessels

    img_name = os.path.split(ct_filepath)[-1].split('.nii')[0]
    vessel_name = output_path + img_name + "_vessel_only_mask"
    overlay_name = output_path + img_name + "_vessels"

    ct_img = nib.load(ct_filepath)
    pixdim = find_pix_dim(ct_img)
    ct_numpy = ct_img.get_fdata()
    ct_numpy = ct_numpy[:, :, 256]
    # print(ct_numpy.min(), ct_numpy.max())

    contours = intensity_seg(ct_numpy, -1000, -300)

    lungs_contour = find_lungs(contours)
    lung_mask = create_mask_from_polygon(ct_numpy, lungs_contour)

    lung_area = compute_area(lung_mask, find_pix_dim(ct_img))

    vessels_only = create_vessel_mask(lung_mask, ct_numpy, denoise=True)

    overlay_plot(ct_numpy, vessels_only)
    plt.title('Overlayed plot')
    plt.savefig(overlay_name)
    plt.close('all')

    save_nifty(vessels_only, vessel_name, affine=ct_img.affine)

    vessel_area = compute_area(vessels_only, find_pix_dim(ct_img))
    ratio = (vessel_area / lung_area) * 100
    print(img_name, 'Vessel %:', ratio)
    lung_areas_csv.append([img_name, lung_area, vessel_area, ratio])
    ratios.append(ratio)

    # Save data to csv file
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(lung_areas_csv)


if __name__ == "__main__":
    # os.makedirs("Predict/output", exist_ok=True)
    # segment_lungs()
    segment_vessels()
