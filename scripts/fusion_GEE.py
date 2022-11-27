# -*- coding: utf-8 -*-
"""
This script performs GEE-based SAR-Optical fusing
It retrieves optical and SAR satellite images from GEE within a period and an AOI,
reconstructs cloudy optical images with corresponding SAR images which are not affected by cloud,
and output infilled optical images

The step numbering refers to the numbered steps provided in the Flowchart
and this flowchart is part of the read_me file.
"""

import GEE_funcs
from utilities import check_task_status
import utilities
import os
import time
import math
import json
import geopandas as gpd
import pandas as pd
import ee
# ee.Authenticate()
ee.Initialize()

#%%

def main():
    """
    ==============================
     Read input and set parameters
    ==============================
    """
    cfg = json.load(open(r"..\config\Parameters.json", 'r'))
    AOI_PATH = cfg["AOI_PATH"]
    AOIs = gpd.read_file(AOI_PATH)

    """
    Setting environment parameters
    """
    # Title of project, used to create output folder on GDrive and GEE asset
    PROJECT_TITLE = cfg["PROJECT_TITLE"]

    # User name of Google Earth Engine, it has to be consistent with the
    # Authenticated Google account
    GEE_USERNAME = cfg["GEE_USERNAME"]

    # Export input data for test out of GEE or not
    EXPORT_INPUT = cfg["EXPORT_INPUT"]

    # Save intermediate outputs or not
    # Save these intermediate outputs can improve the efficiency and
    # avoid exceeding GEE capability
    # Export result of robust linear regression, highly recommended
    EXPORT_RLR = cfg["EXPORT_RLR"]
    # Export result of temporal PCA analysis, set to False if PCA_SMOOTH is
    # False
    EXPORT_PCA = cfg["EXPORT_PCA"]

    # Clear existing intermediate outputs or not
    CLEAR_EXISTING = cfg["CLEAR_EXISTING"]

    """
    Define input parameters
    """

    # Name of Input optical missions, choose from L8 (Landsat8) and S2
    # (Sentinel 2)
    OPTICAL_MISSION = cfg["OPTICAL_MISSION"]
    START_DATE = cfg["START_DATE"]  # Starting date of analysis
    END_DATE = cfg["END_DATE"]  # Ending date of analysis

    # Path of existing metadata
    PATH_MATADATA = cfg["PATH_MATADATA"]  # Default is empty

    # The ratio used to split training and testing (1 means all data for
    # training)
    TRAIN_TEST_SPLIT = cfg["TRAIN_TEST_SPLIT"]
    # Ration of valdation dataset which is split from training dataset with
    # artificial cloud mask
    VAL_SPLIT = cfg["VAL_SPLIT"]
    # Whether split randomly, True=Random, False=Ascending
    RANDOM_SPLIT = cfg["RANDOM_SPLIT"]
    # Whether implementing temporal PCA smooth, set to False when dealing with
    # large memory
    PCA_SMOOTH = cfg["PCA_SMOOTH"]
    # The ratio of PCA components used for reconstructing
    PCA_COMPONENT_RATIO = cfg["PCA_COMPONENT_RATIO"]
    # Threshold of cloud percentage, above which spatial stardardization will
    # be applied
    STD_CLOUD_THRESHOLD = cfg["STD_CLOUD_THRESHOLD"]

    # Area of interest (AOI) should be saved into a json or shp file
    AOI_NAME = cfg["AOI_NAME"]

    print("Project:{} started".format(PROJECT_TITLE))
    nameSuffix = "_{}_{}_{}".format(OPTICAL_MISSION, START_DATE, END_DATE)

    """
    ====================================
    Manage GEE assets
    You can check these assets on
    https://code.earthengine.google.com/
    ====================================
    """
    parent_folder = "projects/earthengine-legacy/assets/users/" + GEE_USERNAME
    subasset_folder = parent_folder + '/' + PROJECT_TITLE
    assets = ee.data.listAssets({'parent': parent_folder})['assets']
    asset_names = [asset['name'].split('/')[-1] for asset in assets]

    if PROJECT_TITLE in asset_names:
        if CLEAR_EXISTING:
            subassets = ee.data.listAssets(
                {'parent': subasset_folder})['assets']
            for subasset in subassets:
                ee.data.deleteAsset(subasset['name'])
            ee.data.deleteAsset(subasset_folder)
            ee.data.createAsset({'type': 'Folder'}, subasset_folder)
    else:
        ee.data.createAsset({'type': 'Folder'}, subasset_folder)

    """
    ==========================
     Preprocess optical images
    ==========================
    """
    AOI = list(AOIs.loc[AOIs['Name'] == AOI_NAME, 'geometry'])[0]
    AOI = ee.Geometry.Polygon(list(AOI.exterior.coords))

    # Dependant variables used for modelling
    dep_variables = ['NDVI']

    # Independant variables used for modelling
    indep_variables = ['VV_mean', 'VH_mean', 'VV_diff', 'VH_diff',
                       'VV_Nmean', 'VH_Nmean', 'VV_Ndiff', 'VH_Ndiff']

    if OPTICAL_MISSION not in ['L8', 'S2']:
        print('Invalid Mission name, please select either S2 or L8')
        raise utilities.InvalidInputs

    # Optical image collection
    collection_ids = {
        'L8': "LANDSAT/LC08/C01/T1_SR",
        'S2': "COPERNICUS/S2_SR"
    }

    # Optical bands used to calculate NDVI
    optical_bands = {
        'L8': ['B4', 'B5', 'pixel_qa'],
        'S2': ['B4', 'B8', 'SCL']
    }

    # Read optical image collection
    optical_collection = ee.ImageCollection(
        collection_ids[OPTICAL_MISSION]).filterBounds(AOI) .select(
            optical_bands[OPTICAL_MISSION], [
                'Red', 'NIR', 'cloud']) .filterDate(
                    START_DATE, END_DATE)

    # Step 1~2, calculate NDVI and preprocess optical images
    optical_collection = GEE_funcs.prepare_optical(optical_collection, AOI,
                                                   OPTICAL_MISSION)
    # Only use images with more than 100 pixels within the AOI
    optical_collection = optical_collection.filterMetadata(
        'PIXEL_COUNT_AOI', 'greater_than', 100)

    if EXPORT_INPUT:
        """Export input NDVI images for SENAP test purpose
        Cloud mask is the same as the output NDVI cloud mask
        """
        # Convert imagecollection to multiband images
        input_NDVI = optical_collection.select('NDVI').toBands()  # Input NDVIs
        collection_time = optical_collection.aggregate_array(
            'system:time_start')
        ID = optical_collection.aggregate_array('system:index')
        indices = ee.List.sequence(0, optical_collection.size().subtract(1))

        def collect_metadata(index):
            """Collection metadata of a image into a feature"""
            dictionary = ee.Dictionary({
                'ID': ID.get(index),
                'CollectionTime': collection_time.get(index)})
            feature = ee.Feature(None, dictionary)
            return feature

        metadata_collection = ee.FeatureCollection(
            indices.map(collect_metadata))

        task1 = utilities.export_image_todrive(
            input_NDVI, AOI,
            'NDVI_Input' + nameSuffix,
            PROJECT_TITLE,
            description='Input NDVI')

        task2 = utilities.export_table_todrive(
            metadata_collection,
            'InputNDVI_Metadata' + nameSuffix,
            PROJECT_TITLE,
            description='Metadata')

        check_task_status(task1)
        check_task_status(task2)

    """
    ==========================
     Preprocess SAR images
    ==========================
    """
    # Read SAR image collection
    S1 = ee.ImageCollection('COPERNICUS/S1_GRD').select(
        [
            'VV',
            'VH']) .filter(
        ee.Filter.listContains(
            'transmitterReceiverPolarisation',
            'VV')) .filter(
        ee.Filter.listContains(
            'transmitterReceiverPolarisation',
            'VH')) .filter(
        ee.Filter.eq(
            'instrumentMode',
            'IW')) .filterBounds(AOI).filterDate(
        START_DATE,
        END_DATE)

    if EXPORT_INPUT:
        """Export input SAR images for SENAP test purpose
        CollectionTime of images can be parsed from image id
        """
        # Convert imagecollection to multiband images
        InputVV = S1.select('VV').toBands()  # Input VV band
        InputVH = S1.select('VH').toBands()  # Input VH band

        collection_time = S1.aggregate_array('system:time_start')
        ID = S1.aggregate_array('system:index')
        indices = ee.List.sequence(0, S1.size().subtract(1))

        def collect_metadata(index):
            """Collection metadata of a image into a feature"""
            dictionary = ee.Dictionary({
                'ID': ID.get(index),
                'CollectionTime': collection_time.get(index)})
            feature = ee.Feature(None, dictionary)
            return feature

        metadata_collection = ee.FeatureCollection(
            indices.map(collect_metadata))

        task1 = utilities.export_image_todrive(
            InputVV, AOI,
            'VV_Input' + nameSuffix,
            PROJECT_TITLE,
            description='Input VV')
        task2 = utilities.export_image_todrive(
            InputVH, AOI,
            'VH_Input' + nameSuffix,
            PROJECT_TITLE,
            description='Input VH')
        task3 = utilities.export_table_todrive(
            metadata_collection, AOI,
            'InputS1_Metadata' + nameSuffix,
            PROJECT_TITLE,
            description='Metadata')

        check_task_status(task1)
        check_task_status(task2)
        check_task_status(task3)

    """
    ==========================
     Pairing and partitioning
    ==========================
    """

    # Step 3, 4 and 5
    opt_SAR = GEE_funcs.pair_opt_SAR(
        optical_collection, S1, AOI, indep_variables)

    # Raise error is there is less than 10 image pairs for model training
    pair_size = opt_SAR.size().getInfo()
    print("Pairing {} and S1".format(OPTICAL_MISSION))
    print("There are {} number of optical-SAR image pairs for analysis".format(pair_size))
    if pair_size < 10:
        print(
            'There are less than 10 images for training, please select a longer time window')
        raise utilities.InvalidInputs

    """
    ================================
    Data split
    ================================
    """
    opt_SAR = ee.ImageCollection(opt_SAR.randomColumn())

    def createConstantBand(img):
        # Creat a constant band with value 1 for multi linear regression
        img = img.addBands(ee.Image(1).rename(['constant']))
        return img.toFloat()

    opt_SAR = ee.ImageCollection(opt_SAR.map(createConstantBand))

    if PATH_MATADATA is None:
        # Set train_test_val property
        if not RANDOM_SPLIT:
            split_position = ee.Dictionary(
                opt_SAR.reduceColumns(
                    ee.Reducer.percentile(
                        [TRAIN_TEST_SPLIT]), ['system:time_start'])
            ).get('p' + str(TRAIN_TEST_SPLIT))

        def set_split_label(img):
            if RANDOM_SPLIT:
                train_test_label = ee.Algorithms.If(
                    ee.Number(img.get('random')).lt(TRAIN_TEST_SPLIT),
                    'Training',
                    'Testing'
                )
            else:
                train_test_label = ee.Algorithms.If(
                    ee.Number(img.get('system:time_start')).lt(split_position),
                    'Training',
                    'Testing'
                )
            split_label = ee.Algorithms.If(
                ee.String(train_test_label).equals('Training'),
                ee.Algorithms.If(
                    ee.Number(img.get('random')).lt(VAL_SPLIT),
                    'Validation',
                    'Training'
                ),
                'Testing'
            )
            return img.set({'Split_label': split_label})

        opt_SAR = opt_SAR.map(set_split_label)

    else:
        metadata = pd.read_csv(PATH_MATADATA)

        Random_dict = ee.Dictionary.fromLists(
            list(
                metadata['ID']), list(
                metadata['Random']))
        SplitLabel_dict = ee.Dictionary.fromLists(
            list(
                metadata['ID']), list(
                metadata['SplitLabel']))
        Cloud_dict = ee.Dictionary.fromLists(
            list(
                metadata['ID']), list(
                metadata['Cloud_cover']))

        def set_prop(img):
            ID = ee.String(img.id())
            img = img.set({
                'random': Random_dict.get(ID),
                'Split_label': SplitLabel_dict.get(ID),
                'CLOUD_PERCENTAGE_AOI0': Cloud_dict.get(ID)
            })
            return img
        opt_SAR = opt_SAR.map(set_prop)

    # Retrieve the artificial mask from existing training data
    art_mask = opt_SAR.filter(
        ee.Filter.rangeContains(
            "CLOUD_PERCENTAGE_AOI", 50, 55
        )
    ).first().select('Mask').multiply(2)

    def add_art_mask(img):
        # Apply artificial mask to validation dataset
        new_mask = ee.Algorithms.If(
            ee.String(img.get('Split_label')).equals('Validation'),
            img.select('Mask').add(art_mask).toFloat(),
            img.select('Mask')
        )
        img = ee.Image(img).addBands(
            ee.Image(new_mask).rename('Mask'),
            overwrite=True)
        return img

    opt_SAR = opt_SAR.map(add_art_mask)

    # Split training and testing
    opt_SAR_train = opt_SAR.filterMetadata(
        'Split_label', 'not_equals', 'Testing'
    )

    """
    ==============================
    Data masking
    ==============================
    """

    # Mask cloud for training
    def mask_cloud(img):
        # Mask cloudy pixels to create clear images for training purpose
        return img.updateMask(img.select('Mask').eq(0)).toFloat()

    opt_SAR_train = opt_SAR_train.map(mask_cloud).select(
        ['constant'] + indep_variables + dep_variables)

    """
    ==============================
    Spatio-temporal modelling
    ==============================
    """

    """
    Step6: Compute robust least squares regression coefficients
    This method in GEE uses iteratively reweighted least squares with the Talwar cost function.
    A point is considered an outlier if the RMS of residuals is greater than beta.
    """
    robust_linear_regression = opt_SAR_train.reduce(
        ee.Reducer.robustLinearRegression(
            numX=len(indep_variables) + 1, numY=1))

    # The results are array images that must be flattened.
    # These lists label the information along each axis of the arrays.
    band_names = [['constant'] + indep_variables, ['NDVI']]

    # Robust linear regression (rlr) coefficients, saved as a multiband image
    rlr_image = robust_linear_regression.select(['coefficients'])\
        .arrayFlatten(band_names).rename(['constant'] + indep_variables)

    if EXPORT_RLR:
        """Trained model is exported to GEE asset to save the previous computation before
        the next heavy computation to avoid exceeding GEE's computation power"""
        subassets = ee.data.listAssets({'parent': subasset_folder})['assets']
        subasset_names = [asset['name'].split('/')[-1] for asset in subassets]
        rlrname = 'rlr_image' + nameSuffix
        rlrID = subasset_folder + '/' + rlrname

        if rlrname not in subasset_names:
            task = utilities.export_image_toasset(
                rlr_image, AOI, rlrID, description='rlr_image')
            check_task_status(task)
        rlr_image = ee.Image(rlrID).select(['constant'] + indep_variables)

    def MLR_predict(img):
        """ Step7: Apply trained Multiple Linear Regression (MLR) model for prediction"""
        NDVI_pred = img.select(
            ['constant'] +
            indep_variables).multiply(
            rlr_image.rename(
                ['constant'] +
                indep_variables)) .reduce('sum').rename('NDVI_pred')
        img = img.select(['NDVI', 'Mask']).addBands(NDVI_pred)
        return img

    opt_SAR_outputs = opt_SAR.map(MLR_predict).select(
        ['NDVI', 'NDVI_pred', 'Mask'])

    """
    ================================================================
    Post process including PCA smoothing and spatial stardardization
    ================================================================
    """
    # Step 8
    if PCA_SMOOTH:
        NDVI_smoothed = GEE_funcs.Temporal_PCA(
            opt_SAR_outputs.select('NDVI_pred'),
            AOI,
            opt_SAR_outputs.size().multiply(PCA_COMPONENT_RATIO).floor().int(),
            10)
    else:
        NDVI_smoothed = opt_SAR_outputs.select('NDVI_pred').toBands()

    if EXPORT_PCA:
        """Smoothed NDVI images are exported to GEE asset to save the previous computation before
        the next heavy computation to avoid exceeding GEE's computation power"""
        if OPTICAL_MISSION == 'S2':
            S2_ids = NDVI_smoothed.bandNames()

            def cat_S2_id(element):
                """Concat S2 ids with S2 as initials"""
                return ee.String('S2_').cat(element)
            S2_ids_new = S2_ids.map(cat_S2_id)
            NDVI_smoothed = NDVI_smoothed.select(S2_ids, S2_ids_new)

        subassets = ee.data.listAssets({'parent': subasset_folder})['assets']
        subasset_names = [asset['name'].split('/')[-1] for asset in subassets]
        smoothed_name = 'smoothedImage' + nameSuffix
        smoothed_id = subasset_folder + '/' + smoothed_name

        if smoothed_name not in subasset_names:
            task = utilities.export_image_toasset(
                NDVI_smoothed,
                AOI,
                smoothed_id,
                description='Smoothed NDVI')
            check_task_status(task)

        NDVI_smoothed = ee.Image(smoothed_id)
        if OPTICAL_MISSION == 'S2':
            NDVI_smoothed = NDVI_smoothed.select(S2_ids_new, S2_ids)

    # Step 9&10
    NDVI_calibrated, NDVI_filled = GEE_funcs.post_process(
        opt_SAR_outputs, NDVI_smoothed, AOI, STD_CLOUD_THRESHOLD
    )

    """
    ====================================================
    Export output images to GDrive and metadata to local
    ====================================================
    """

    """Export images to GDrive"""

    # Convert imagecollection to multiband images
    OutputFilled = NDVI_filled.select(
        'NDVI').toBands().clip(AOI)  # Output NDVIs
    OutputMask = NDVI_filled.select(
        'Mask').toBands().clip(AOI)  # Output Cloud Mask
    OutputPred = NDVI_calibrated.select('NDVI_pred').toBands().clip(AOI)
    OutputObs = NDVI_calibrated.select('NDVI').toBands().clip(AOI)

    """Export metadata to drive"""

    cloud_cover = NDVI_filled.aggregate_array('CLOUD_PERCENTAGE_AOI')
    collection_time = NDVI_filled.aggregate_array('system:time_start')
    ID = NDVI_filled.aggregate_array('system:index')
    MAE = NDVI_filled.aggregate_array('MAE')
    Random = NDVI_filled.aggregate_array('random')
    SplitLabel = NDVI_filled.aggregate_array('Split_label')

    indices = ee.List.sequence(0, NDVI_filled.size().subtract(1))

    def collect_metadata(index):
        """Collection metadata of a image into a feature"""
        dictionary = ee.Dictionary({
            'ID': ID.get(index),
            'Cloud_cover': cloud_cover.get(index),
            'CollectionTime': collection_time.get(index),
            'Random': Random.get(index),
            'MAE': MAE.get(index),
            'SplitLabel': SplitLabel.get(index)})
        feature = ee.Feature(None, dictionary)
        return feature

    metadata_collection = ee.FeatureCollection(indices.map(collect_metadata))

    task1 = utilities.export_image_todrive(
        OutputPred, AOI,
        'NDVI_Pred' + nameSuffix,
        PROJECT_TITLE,
        description='NDVI prediction')
    task2 = utilities.export_image_todrive(
        OutputObs, AOI,
        'NDVI_Obs' + nameSuffix,
        PROJECT_TITLE,
        description='NDVI obs')

    task3 = utilities.export_image_todrive(
        OutputFilled, AOI,
        'NDVI_gapfree' + nameSuffix,
        PROJECT_TITLE,
        description='NDVI gap filled')
    task4 = utilities.export_image_todrive(
        OutputMask, AOI,
        'NDVI_Mask' + nameSuffix,
        PROJECT_TITLE,
        description='NDVI cloud mask')

    task5 = utilities.export_table_todrive(
        metadata_collection,
        'Output_Metadata' + nameSuffix,
        PROJECT_TITLE,
        description='Metadata')

    check_task_status(task1, cancel_when_interrupted=False)
    check_task_status(task2, cancel_when_interrupted=False)
    check_task_status(task3, cancel_when_interrupted=False)
    check_task_status(task4, cancel_when_interrupted=False)
    check_task_status(task5, cancel_when_interrupted=False)


if __name__ == "__main__":
    main()
