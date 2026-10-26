# -*- coding: utf-8 -*-
"""
@author: Yazdan, salimiyazdan@gmail.com
"""
import SimpleITK as sitk
import monai
import numpy as np
from termcolor import cprint
import torch
from tqdm import tqdm
import os
import nibabel as nib
import gc
from natsort import os_sorted
from glob import glob
         
           
def match_space(input_image, reference_image, interpolate = "linear", DefaultPixelValue = 0):
    if isinstance(input_image, str):
        input_image = sitk.ReadImage(input_image)
    if isinstance(reference_image, str):
        reference_image = sitk.ReadImage(reference_image)
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(reference_image.GetSpacing())
  
    resampler.SetSize(reference_image.GetSize())
    resampler.SetOutputOrigin(reference_image.GetOrigin())
    resampler.SetOutputDirection(reference_image.GetDirection())
    # Set the default pixel value to -1000
    resampler.SetDefaultPixelValue(DefaultPixelValue)
    if interpolate == "linear":
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    elif interpolate == "nearest":
        resampler.SetInterpolator(sitk.sitkLinear)
    elif interpolate.lower() == "bspline":
        resampler.SetInterpolator(sitk.sitkBSpline)
    resampled_image = resampler.Execute(input_image)
    return resampled_image


def data_loader(list_input, list_output, 
                cache_rate = 1, augmentation = False, transform = "default",
                batch_size = 16, pixdim = (1,1,1), patch_size = (256,256,1),
                orientation = "RAS", input_interpolation = "bilinear", 
                output_interpolation = "bilinear", samrt_num_workers = 0, 
                num_samples = 30, positive = 1, negative =1 , input_lower_intensity = -70, 
                input_upper_intensity = 170, output_lower_intensity = 0,
                output_upper_intensity = 1,
                input_b_min = 0, input_b_max = 1, 
                output_b_min = 0, output_b_max = 1,
                randcroppad_tresh = 0, ImageCropForeGroundMargin = 0, 
                SegmentCropForeGroundMargin = (20,20,40),
                clip = True, task = "segmentation", data_split = "train", 
                data_loader_type = "Thread", cachedata = True,
                data_loader_num_threads = 0,
                scale_image_to_self_max = False,
                num_segment_classes = 2,
                segmentation_threshold=0.5,
                ):

    def do_nothing(input_value):
        return input_value
    ImageCropForeGroundMargin = (ImageCropForeGroundMargin,) * len(pixdim)
    data_dictionary = [{"input_image": input_image, "output_image": output_image} 
                        for input_image, output_image  in 
                        zip(list_input,list_output)]
    if transform == "default":
        if data_split == "train":
            if augmentation and task == "regression":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, output_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation)if orientation != "keep" else monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                        # monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin),

                        monai.transforms.ScaleIntensityRanged(keys = "output_image", 
                                                              a_min = output_lower_intensity, a_max = output_upper_intensity,
                                                              clip=clip, b_min = output_b_min, b_max = output_b_max) if not scale_image_to_self_max else monai.transforms.ScaleIntensityd(keys = ["output_image"],
                                                                                                    minv = output_lower_intensity, maxv=output_upper_intensity),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity,
                                                              b_min = input_b_min, b_max = input_b_max),
                        monai.transforms.RandCropByPosNegLabeld(keys=["input_image", "output_image"],
                                                                label_key = "output_image", 
                                                                spatial_size = patch_size,
                                                                allow_smaller = True, 
                                                                image_key= "input_image", num_samples=num_samples,
                                                                pos = positive, neg = negative),
                        monai.transforms.ResizeWithPadOrCropd(keys=["input_image", "output_image"],
                                                              spatial_size= patch_size),
                        monai.transforms.RandFlipd(keys=["input_image", "output_image"]), # add augmentations after all processes
                        
                    ]
                )
            elif augmentation and task == "segmentation":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, "nearest"]),
                        # monai.transforms.AsDiscreted(keys=["output_image"], to_onehot=2), 
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin, allow_smaller=True),
                        # monai.transforms.DataStatsd(keys=["output_image"]),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity, 
                                                              a_max = input_upper_intensity,
                                                              clip=clip,  b_min = input_b_min, b_max = input_b_max),
                        monai.transforms.RandShiftIntensityd(keys = "input_image", offsets = (.95,1.05)),
                        monai.transforms.RandGaussianNoised(keys = "input_image"),
                        monai.transforms.RandAdjustContrastd(keys = "input_image"),
                        monai.transforms.RandHistogramShiftd(keys = "input_image"),
                        # monai.transforms.RandZoom(keys = "input_image"),
                        monai.transforms.RandScaleIntensityd(keys = "input_image", factors=(.98,1.02)),
                        monai.transforms.RandCropByPosNegLabeld(keys=["input_image", "output_image"],
                                                                label_key = "output_image", 
                                                                spatial_size = patch_size,
                                                                allow_smaller = True, 
                                                                image_key= "input_image", num_samples=num_samples,
                                                                pos = positive, neg = negative, image_threshold = randcroppad_tresh),
                        monai.transforms.ResizeWithPadOrCropd(keys=["input_image", "output_image"],
                                                              spatial_size= patch_size),
                        monai.transforms.RandFlipd(keys=["input_image", "output_image"]), # add augmentations after all processes

                    ]
                )
                post_transform = "none"
                    
            elif not augmentation and task == "regression":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, output_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation) if orientation != "keep" else monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                        # monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin),
                        
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity,
                                                              a_max = input_upper_intensity, 
                                                              b_min = input_b_min, b_max = input_b_max, clip=clip),
                        monai.transforms.ScaleIntensityRanged(keys = "output_image", a_min = output_lower_intensity,
                                                              a_max = output_upper_intensity, 
                                                              b_min = output_b_min, b_max = output_b_max, clip=clip) if not scale_image_to_self_max else monai.transforms.ScaleIntensityd(keys = ["output_image"],
                                                                                                    minv = output_lower_intensity, maxv=output_upper_intensity),
                        monai.transforms.RandCropByPosNegLabeld(keys=["input_image", "output_image"],
                                                                label_key = "output_image", 
                                                                spatial_size = patch_size,
                                                                allow_smaller = True, 
                                                                image_key= "input_image", num_samples=num_samples,
                                                                pos = positive, neg = negative),
                        monai.transforms.ResizeWithPadOrCropd(keys=["input_image", "output_image"],
                                                              spatial_size= patch_size),
                        
                    ]
                )
                post_transform = "none"
            elif not augmentation and task == "segmentation":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, "nearest"]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                        # monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity, a_max = input_upper_intensity,
                                                              clip=clip, b_min = input_b_min, b_max = input_b_max),
                        monai.transforms.RandCropByPosNegLabeld(keys=["input_image", "output_image"],
                                                                label_key = "output_image", 
                                                                spatial_size = patch_size,
                                                                allow_smaller = True, 
                                                                image_key= "input_image", num_samples=num_samples,
                                                                pos = positive, neg = negative),
                        monai.transforms.ResizeWithPadOrCropd(keys=["input_image", "output_image"],
                                                              spatial_size= patch_size),
                        
                    ]
                )
                post_transform = monai.transforms.Compose(
                    [
                        monai.transforms.AsDiscreted(keys="predict_image", argmax=True, to_onehot=2),
                        monai.transforms.AsDiscreted(keys="output_image", to_onehot=2),
                    ]
                )
        elif data_split == "test":
            if task == "regression":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image"], pixdim = pixdim,
                                                  mode = [input_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image"], axcodes=orientation) if orientation != "keep" else monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.CropForegroundd(keys = ["input_image"], source_key = "input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                        monai.transforms.ScaleIntensityRanged(keys = ["input_image"], a_min = input_lower_intensity,
                                                              a_max = input_upper_intensity, 
                                                              b_min = input_b_min, b_max = input_b_max, clip=clip),
                        # monai.transforms.ScaleIntensityRanged(keys = "output_image", a_min = output_lower_intensity, a_max = output_upper_intensity, 
                        #                                       b_min = output_b_min, b_max = output_b_max, clip=clip),
                    ]
                )
                
                post_transform = monai.transforms.Compose(
                    [
                        monai.transforms.Invertd(
                            keys="predict_image",
                            transform=initial_transform,
                            orig_keys="input_image",
                            # meta_keys="predict_image_meta_dict",
                            # orig_meta_keys="input_image_meta_dict",
                            meta_key_postfix="meta_dict",
                            nearest_interp=False,
                            to_tensor=True,
                            device="cpu",
                            allow_missing_keys = False,
                        ),
                        monai.transforms.ScaleIntensityRanged(keys = ["predict_image"], a_min = output_b_min,
                                                              a_max = output_b_max, 
                                                              b_min = output_lower_intensity, b_max = output_upper_intensity, clip=clip), 
                        # monai.transforms.ScaleIntensityRanged(keys = ["predict_image"], a_min = input_lower_intensity,
                        #                                       a_max = input_upper_intensity, 
                        #                                       b_min = input_b_min, b_max = input_b_max, clip=clip), 
                    ]
                )
            elif task == "segmentation":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image"], pixdim = pixdim,
                                                  mode = [input_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image"], axcodes=orientation),
                        # monai.transforms.CropForegroundd(keys=["input_image"], source_key="input_image", margin = ImageCropForeGroundMargin),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity, a_max = input_upper_intensity,
                                                              clip=clip, b_min = input_b_min, b_max = input_b_max),                            
                    ]
                )
                post_transform = monai.transforms.Compose(
                    [
                        monai.transforms.Invertd(
                            keys="predict_image",
                            transform=initial_transform,
                            orig_keys="input_image",
                            meta_keys="predict_image_meta_dict",
                            orig_meta_keys="input_image_meta_dict",
                            meta_key_postfix="meta_dict",
                            nearest_interp=False,
                            to_tensor=True,
                            device="cpu",
                            allow_missing_keys = True,
                        ),
                        monai.transforms.Activationsd(keys="predict_image", sigmoid=True, squared_pred = False),
                        monai.transforms.AsDiscreted(keys="predict_image", to_one_hot = num_segment_classes, threshold = segmentation_threshold),
                    ]
                )
        elif data_split == "validation":
            if task == "regression":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, output_interpolation]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation) if orientation != "keep" else monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        
                        # monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity,
                                                              a_max = input_upper_intensity, 
                                                              b_min = input_b_min, b_max = input_b_max, clip=clip),
                        monai.transforms.ScaleIntensityRanged(keys = "output_image", a_min = output_lower_intensity, a_max = output_upper_intensity, 
                                                              b_min = output_b_min, b_max = output_b_max, clip=clip),
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                    ]
                )
                
                post_transform = "none"
            elif task == "segmentation":
                initial_transform = monai.transforms.Compose(
                    [
                        monai.transforms.LoadImaged(keys=["input_image", "output_image"],
                                                    ensure_channel_first = True, image_only = False),
                        monai.transforms.Spacingd(keys=["input_image", "output_image"], pixdim = pixdim,
                                                  mode = [input_interpolation, "nearest"]),
                        monai.transforms.EnsureTyped(keys=["input_image", "output_image"]),
                        monai.transforms.Orientationd(keys=["input_image", "output_image"], axcodes=orientation),
                        # monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="output_image", margin = SegmentCropForeGroundMargin),
                        monai.transforms.ScaleIntensityRanged(keys = "input_image", a_min = input_lower_intensity, a_max = input_upper_intensity,
                                                              clip=clip, b_min = input_b_min, b_max = input_b_max),         
                        monai.transforms.CropForegroundd(keys=["input_image", "output_image"], source_key="input_image", margin = ImageCropForeGroundMargin, allow_smaller=True),
                    ]
                )
                post_transform = monai.transforms.Compose(
                    [
                        monai.transforms.Activations(sigmoid=True),
                        monai.transforms.AsDiscreted(keys="predict_image", argmax = True, to_onehot = num_segment_classes),
                        monai.transforms.AsDiscreted(keys="output_image", to_onehot = num_segment_classes, threshold = segmentation_threshold),
                    ]
                )
    else: # not default transform
        initial_transform = transform
        if task == "regression" and data_split == "test":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.Invertd(
                        keys="predict_image",
                        transform=initial_transform,
                        orig_keys="input_image",
                        meta_keys="predict_image_meta_dict",
                        orig_meta_keys="input_image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cpu",
                    ),
                ]
            )
        elif task == "segmentation" and data_split == "test":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.Invertd(
                        keys="predict_image",
                        transform=initial_transform,
                        orig_keys="input_image",
                        meta_keys="predict_image_meta_dict",
                        orig_meta_keys="input_image_meta_dict",
                        meta_key_postfix="meta_dict",
                        nearest_interp=False,
                        to_tensor=True,
                        device="cpu",
                    ),
                    monai.transforms.Activationsd(keys="predict_image", sigmoid=True, squared_pred = True),
                    monai.transforms.AsDiscreted(keys="predict_image", to_one_hot = num_segment_classes, threshold = segmentation_threshold),
                ]
            )
        elif task == "regression" and data_split == "validation" or "train":
            post_transform = "none"
        elif task == "segmentation" and data_split == "validation" or "train":
            post_transform = monai.transforms.Compose(
                [
                    monai.transforms.AsDiscreted(keys="predict_image", argmax=True, to_onehot=num_segment_classes, threshold = segmentation_threshold),
                    monai.transforms.AsDiscreted(keys="output_image", to_onehot=num_segment_classes, threshold = segmentation_threshold),
                ]
            )
            
    if cachedata:
        # initial_dataset = monai.data.SmartCacheDataset(
        #     data = data_dictionary,
        #     transform = initial_transform,
        #     cache_rate = cache_rate,
        #     num_init_workers = samrt_num_workers,
        #     num_replace_workers = samrt_num_workers,
        #     copy_cache=True, shuffle = True,
        # )
        initial_dataset = monai.data.CacheDataset(
            data = data_dictionary,
            transform = initial_transform,
            cache_rate = cache_rate,
            num_workers = samrt_num_workers,
            copy_cache=True, 
        )
    else:
        initial_dataset = monai.data.Dataset(
            data = data_dictionary,
            transform = initial_transform,
        )
        
    if data_loader_type == "Thread":
        data_loader = monai.data.ThreadDataLoader(
            initial_dataset,
            batch_size = batch_size,
            num_workers = data_loader_num_threads,
            pin_memory = True,
            shuffle = True,
            use_thread_workers = bool(data_loader_num_threads),
            persistent_workers = bool(data_loader_num_threads),
        )
    else:
        data_loader = monai.data.DataLoader(
            initial_dataset,
            batch_size = batch_size,
            num_workers = data_loader_num_threads,
            pin_memory = True,
            shuffle = True,
            persistent_workers = bool(data_loader_num_threads),
        )
    
    check_data = monai.utils.misc.first(data_loader)
    input_shape = check_data["input_image"].shape
    output_shape = check_data["output_image"].shape  
    input_sample = np.squeeze(check_data["input_image"].numpy())
    output_sample = np.squeeze(check_data["output_image"].numpy())
    
    return_dictionary = {"data_loader" : data_loader, "initial_transform":initial_transform,
                         "input_sample":input_sample,"input_shape":input_shape, "output_sample":output_sample,
                         "output_shape":output_shape, "post_transform":post_transform}
    return return_dictionary



def ensemble_image_regression(list_inference_models, target_url = "none"):
   
    image_ensemble = sitk.ReadImage(list_inference_models[0])
    for url in list_inference_models[1:]:
        try:
            image_ensemble += sitk.ReadImage(url)
        except:
            image_ensemble += sitk.Cast(match_space(input_image = url, reference_image = image_ensemble), image_ensemble.GetPixelID())
            cprint(f"\n{url} was matched by force!\n", "white", "on_yellow")
    image_ensemble  = image_ensemble / len(list_inference_models)
    if  target_url != "none":
        sitk.WriteImage(image_ensemble, target_url)
    return image_ensemble

 
def model_inference_regression(model_url, list_images="from-model", predict_directory="none",
                               device="auto", prefix="", suffix="", cache_rate=1,
                               sliding_overlap=0.25, samrt_num_workers=8, *,
                               pixdim=None, orientation="keep", input_range=None,
                               output_range=None, sliding_window_shape=(64, 64, 64),
                               normalize_features=True, downsample="merging"):
    """Run AC_Model.pth, SC_Model.pth or ASC_Model.pth on 3D NIfTI images.

    Training preprocessing is absent from these state_dict files. By default,
    keep native spacing/orientation and input intensities; outputs are raw
    network values. input_range=(min,max) maps inputs to [0,1] with clipping;
    output_range=(min,max) maps network output [0,1] back to those units.
    normalize_features and downsample are MONAI defaults, not recovered settings.
    cache_rate and samrt_num_workers remain for compatibility; files run serially.
    """
    from pathlib import Path
    from monai.data import MetaTensor
    from monai.transforms import Compose, LoadImaged, EnsureTyped, Orientationd, Spacingd, Invertd

    if isinstance(list_images, (str, Path)):
        if str(list_images) == "from-model":
            raise ValueError("Pass list_images explicitly; the MAA weights contain no test images.")
        list_images = [list_images]
    list_images = list(list_images)
    if not list_images or predict_directory == "none":
        raise ValueError("Supply input images and predict_directory.")
    roi = tuple(sliding_window_shape)
    if len(roi) != 3 or any(type(n) is not int or n < 32 or n % 32 for n in roi) or np.prod(roi) <= 32**3:
        raise ValueError("sliding_window_shape must use multiples of 32, e.g. (64,64,64), with more than 32**3 voxels.")
    if sliding_overlap == "from-model":
        sliding_overlap = 0.25
    if not 0 <= sliding_overlap < 1:
        raise ValueError("sliding_overlap must be in [0,1).")
    for limits in (input_range, output_range):
        if limits is not None and (len(limits) != 2 or not np.isfinite(limits).all() or limits[1] <= limits[0]):
            raise ValueError("Intensity ranges must contain two finite increasing values.")
    if pixdim is not None and (len(pixdim) != 3 or not np.isfinite(pixdim).all() or any(n <= 0 for n in pixdim)):
        raise ValueError("pixdim must contain three positive finite voxel sizes in mm.")
    if type(normalize_features) is not bool or downsample not in ("merging", "mergingv2"):
        raise ValueError("Use a boolean normalize_features and downsample='merging' or 'mergingv2'.")
    output_dir = Path(predict_directory)
    sources = [Path(p) for p in list_images]
    if any(not (p.name.endswith(".nii.gz") or p.suffix == ".nii") for p in sources):
        raise ValueError("Input images must be .nii or .nii.gz files.")
    names = [f"{prefix}{p.name.removesuffix('.gz').removesuffix('.nii')}{suffix}.nii.gz" for p in sources]
    if any(Path(name).name != name for name in names):
        raise ValueError("prefix and suffix cannot contain directory separators.")
    destinations = [output_dir / name for name in names]
    if len(set(p.resolve() for p in destinations)) != len(destinations):
        raise ValueError("Input names produce duplicate output filenames.")
    if any(p.exists() or p.resolve() in {s.resolve() for s in sources} for p in destinations):
        raise FileExistsError("An output exists or would overwrite an input; choose another directory or suffix.")
    with open(model_url, "rb") as stream:
        if stream.read(64).startswith(b"version https://git-lfs.github.com/spec/v1"):
            raise ValueError("The model is an LFS pointer. Run git lfs pull first.")
    state = torch.load(model_url, map_location="cpu", weights_only=True)
    # All three supplied checkpoints have this same architecture (159 tensors).
    model = monai.networks.nets.SwinUNETR(
        in_channels=1, out_channels=1, feature_size=24, spatial_dims=3,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), patch_size=2,
        window_size=7, norm_name="instance", normalize=normalize_features,
        downsample=downsample, use_v2=False,
    )
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use device='cpu'.")
    model.to(device).eval()
    transforms = [LoadImaged("input_image", reader="NibabelReader", ensure_channel_first=True, mmap=False),
                  EnsureTyped("input_image", dtype=torch.float32)]
    if orientation != "keep":
        transforms.append(Orientationd("input_image", axcodes=orientation, labels=(("L", "R"), ("P", "A"), ("I", "S"))))
    if pixdim is not None:
        transforms.append(Spacingd("input_image", pixdim=pixdim, mode="bilinear"))
    transform = Compose(transforms)
    inverse = Invertd("predict_image", transform=transform, orig_keys="input_image",
                      nearest_interp=False, to_tensor=True, device="cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    for source, destination in tqdm(list(zip(sources, destinations)), desc="MAA inference"):
        original = nib.load(source, mmap=False)
        if len(original.shape) != 3 or not np.isfinite(original.get_fdata(dtype=np.float32)).all():
            raise ValueError(f"Expected a finite 3D NIfTI image: {source}")
        batch = transform({"input_image": str(source)})
        image = batch["input_image"].as_tensor().unsqueeze(0)
        if input_range is not None:
            image = ((image - input_range[0]) / (input_range[1] - input_range[0])).clamp(0, 1)
        if not torch.isfinite(image).all():
            raise ValueError(f"Preprocessing produced nonfinite values: {source}")
        with torch.inference_mode():
            prediction = monai.inferers.sliding_window_inference(
                image, roi, 1, model, overlap=sliding_overlap, mode="gaussian",
                sw_device=device, device="cpu",
            )[0]
        if output_range is not None:
            prediction = prediction * (output_range[1] - output_range[0]) + output_range[0]
        batch["predict_image"] = MetaTensor(prediction)
        restored = inverse(batch)["predict_image"] if batch["input_image"].applied_operations else batch["predict_image"]
        array = restored[0].cpu().numpy().astype(np.float32)
        if array.shape != original.shape or not np.isfinite(array).all():
            raise ValueError(f"Invalid output shape or nonfinite prediction: {source}")
        header = original.header.copy()
        header.set_data_dtype(np.float32)
        header.set_slope_inter(1, 0)
        predicted_nifti = nib.Nifti1Image(array, original.affine, header)
        predicted_nifti.set_qform(*original.get_qform(coded=True))
        predicted_nifti.set_sform(*original.get_sform(coded=True))
        nib.save(predicted_nifti, destination)
    return {"predict_image_to_write": predicted_nifti, "test_batch_data": batch,
            "test_predicted_image": restored, "output_urls": [str(p) for p in destinations]}


def FreeGPU(gc_collect = True, torch_empty_cache = True):
    if gc_collect:
        gc.collect()
    if torch_empty_cache:
        torch.cuda.empty_cache()
        
        
def CopyInfo(ReferenceImage, UpdatingImage, origin = True, spacing = True, direction = True):
    if isinstance(ReferenceImage, str):
        ReferenceImage = sitk.ReadImage(ReferenceImage)
    if isinstance(UpdatingImage, str):
        UpdatingImage = sitk.ReadImage(UpdatingImage)
    UpdatedImage = UpdatingImage 
    if origin:
        UpdatedImage.SetOrigin(ReferenceImage.GetOrigin())
    if spacing:
        UpdatedImage.SetSpacing(ReferenceImage.GetSpacing())
    if direction:
        UpdatedImage.SetDirection(ReferenceImage.GetDirection())
    return UpdatedImage


def sitk_rescale(image, input_min = "image-min", input_max = "image-max", output_min = 0, output_max = 1):
    if isinstance(image, str):
        image = sitk.ReadImage(image)
    image_array = sitk.GetArrayFromImage(image)
    if input_min == "image-min":
        input_min = np.min(image_array)
    if input_max == "image-max":
        input_max = np.max(image_array)
        
    array_no_clip = (image_array - input_min) / (input_max - input_min)
    array_no_clip = output_min + (array_no_clip * (output_max - output_min))
    image_no_clip = sitk.GetImageFromArray(array_no_clip)
    image_no_clip = CopyInfo(ReferenceImage = image, UpdatingImage = image_no_clip)
    
    array_clip = np.interp(image_array, (input_min, input_max), (output_min, output_max))
    image_clip = sitk.GetImageFromArray(array_clip)
    image_clip = CopyInfo(ReferenceImage = image, UpdatingImage = image_clip)
    return image_clip, image_no_clip

def sitk_percentile(image_url, percentile, segment_url = "none"):

    if isinstance(image_url, str):
        image_array = sitk.GetArrayFromImage(sitk.ReadImage(image_url))
    elif isinstance(image_url, sitk.Image):
        image_array = sitk.GetArrayFromImage(image_url)
    elif isinstance(image_url, np.ndarray):
        image_array = image_url
    if segment_url != "none":
        if isinstance(segment_url, str):
            segment_array = sitk.GetArrayFromImage(sitk.ReadImage(segment_url))
        elif isinstance(segment_url, sitk.Image):
            segment_array = sitk.GetArrayFromImage(segment_url)
        elif isinstance(segment_url, np.ndarray):
            segment_array = segment_url
            
        image_array = image_array[segment_array != 0]
    percentile_value_non_zeros = np.percentile(image_array[image_array != 0], percentile)
    percentile_value_with_zeros = np.percentile(image_array, percentile)
    return percentile_value_non_zeros, percentile_value_with_zeros

def ensemble_regression_folds(list_images,
                              model_directory,
                              predict_directory,
                              model_criteria = "all",
                              device = "cuda",
                              folds = "all",
                              normalize_images = False,
                              percentile = 99.,
                              sliding_overlap = "from-model",
                              start_model_number = 0,
                              remove_single_folds = False,
                              ):

    if normalize_images:
        list_normalized_images = []
        for image_url in tqdm(list_images, desc = "normalizing images"):
            image_normalized = sitk_rescale(image_url, 
                                            input_min = "image-min",
                                            input_max = sitk_percentile(image_url, percentile)[0],
                                            output_min = 0, output_max = 1,
                                            )[-1]
            
            sitk.WriteImage(image_normalized, image_url.replace(".nii.gz", "--normalized.nii.gz"))
            list_normalized_images.append(image_url.replace(".nii.gz", "--normalized.nii.gz"))
        list_images = list_normalized_images
    if folds == "all":
        if model_criteria == "all":
            list_models = os_sorted(glob(os.path.join(model_directory, "fold*", "*full.pth")))
        else:
            list_models = os_sorted(glob(os.path.join(model_directory, "fold*", model_criteria)))
            
    else:
        if model_criteria == "all":
            list_models = os_sorted(glob(os.path.join(model_directory, "fold*", "*full.pth")))
        else:
            list_models = os_sorted(glob(os.path.join(model_directory, "fold*", model_criteria)))
        list_models = [x for x in list_models if int(x.split("\\")[-2].split("--")[-1]) in folds]
       
    list_models = list_models[start_model_number:]

    model_names_to_ensemble = []
    for model_url in tqdm(list_models):
        cprint(f"\n{model_url}\n", "white", "on_green")
        fold = model_url.split("\\")[-2]
        model_name_to_suffix = os.path.basename(model_url).replace("Model-Full.pth", "").replace("Model-full.pth", "")
        model_names_to_ensemble.append(model_name_to_suffix)
        model_inference_regression(model_url, 
                                             list_images = list_images,
                                             predict_directory = predict_directory, 
                                             suffix = f"_{fold}_{model_name_to_suffix}",
                                             sliding_overlap = sliding_overlap,
                                             )
        FreeGPU(gc_collect = False, torch_empty_cache = True)
    model_names_to_ensemble = list(set(model_names_to_ensemble))
    for model_name_to_ensemble in tqdm(model_names_to_ensemble, desc = "Ensembling Folds", colour = "cyan"):
        list_ensembled_images = []
        for input_image_url in tqdm(list_images, desc = model_name_to_ensemble, colour = "green"):
            image_name = os.path.basename(input_image_url).replace(".nii.gz", "")
            list_images_to_be_ensembled = glob(os.path.join(predict_directory, f"{image_name}_*{model_name_to_ensemble}.nii.gz"))
            ensemble_image_regression(list_images_to_be_ensembled, 
                                                target_url = os.path.join(predict_directory,
                                                                          f"{image_name}-{model_name_to_ensemble}_Ensemble.nii.gz"))
            list_ensembled_images.append(os.path.join(predict_directory,
                                      f"{image_name}-{model_name_to_ensemble}_Ensemble.nii.gz"))
            
    if remove_single_folds:
        list_predicted_single_folds = os_sorted(glob(os.path.join(predict_directory, "*_fold--*_*.nii.gz")))
        deleted_objects = [os.remove(x) for x in tqdm(list_predicted_single_folds, desc = "Warning! Removeing Single Folds", colour = "red")]
    return list_ensembled_images