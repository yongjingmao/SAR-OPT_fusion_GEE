# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:02:10 2021

@author: uqymao1
"""
import ee
import time
ee.Initialize()


def prepare_optical(optical_collection, AOI, optical_mission):
    """
     Preprocess optical images
    """
    def cal_NDVI(img):
        """Step1 : Calculate NDVI for optical images"""
        NDVI = img.select('NIR').subtract(img.select('Red'))\
            .divide(img.select('NIR').add(img.select('Red'))).rename('NDVI')
        return img.addBands(NDVI)

    def add_cloudMask_L8(img):
        """ Step2: Process Landsat 8 cloud flag
        In the cloud mask, 1 is for cloudy pixel.
        """
        # Bits 3 and 5 are cloud shadow and cloud, respectively.
        cirrus_bitMask = 1 << 2
        cloudshadow_bitMask = 1 << 3
        clouds_bitMask = 1 << 5

        # Select band cloud flag
        qa = img.select('cloud')

        # Both flags should be set to zero, indicating clear conditions.
        mask = qa.bitwiseAnd(cloudshadow_bitMask)\
            .Or(qa.bitwiseAnd(clouds_bitMask))\
            .Or(qa.bitwiseAnd(cirrus_bitMask)).rename('Mask')

        # Calculate cloud percentage as the percentage of cloudy pixels in all
        # pixels
        cloud_sum = ee.Number(
            mask.reduceRegion(
                ee.Reducer.sum(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))
        cloud_count = ee.Number(
            mask.reduceRegion(
                ee.Reducer.count(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))
        cloud_percent = cloud_sum.divide(cloud_count).multiply(100)

        return img.addBands(mask).set({
            'CLOUD_PERCENTAGE_AOI': cloud_percent,
            'PIXEL_COUNT_AOI': cloud_count})

    def add_cloudMask_S2(img):
        """ Step2: Process Sentinel cloud flag
        Number 3 is cloud shadow and above 7 are cloud with different
        confidence respectively.
        In the cloud mask, 1 is for cloudy pixel.
        """
        mask = img.select('cloud').gte(7).Or(
            img.select('cloud').eq(3)).rename('Mask')

        # Calculate cloud percentage as the percentage of cloudy pixels in all
        # pixels
        cloud_sum = ee.Number(
            mask.reduceRegion(
                ee.Reducer.sum(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))
        cloud_count = ee.Number(
            mask.reduceRegion(
                ee.Reducer.count(),
                AOI,
                100,
                maxPixels=1e9).get('Mask'))
        cloud_percent = cloud_sum.divide(cloud_count).multiply(100)

        return img.addBands(mask).set({
            'CLOUD_PERCENTAGE_AOI': cloud_percent,
            'PIXEL_COUNT_AOI': cloud_count})

    if optical_mission == 'L8':
        optical_collection = optical_collection.map(cal_NDVI).map(
            add_cloudMask_L8).select(['NDVI', 'Mask'])
    else:
        optical_collection = optical_collection.map(cal_NDVI).map(
            add_cloudMask_S2).select(['NDVI', 'Mask'])
    return optical_collection


def cal_covariates(img, AOI, indep_variables):
    """
    Step 3 Calculate temporal and spatial covariates with spatial-first method
    """
    # Mask dark edges in S1 images
    img = img.updateMask(img.gt(-40))

    # Step 3.1 calculate temporal covariates
    spatial_mean = img.select(['VV', 'VH'])\
        .reduceRegion(ee.Reducer.mean(), AOI, 100, maxPixels=1e9)\
        .toImage(['VV', 'VH'])
    neighbor_mean = img.select(['VV', 'VH'])\
        .reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.square(10))

    # Step 3.2 calculate spatial covariates
    spatial_diff = img.select(['VV', 'VH'])\
        .subtract(spatial_mean.select(['VV', 'VH']))
    neighbor_diff = img.select(['VV', 'VH'])\
        .subtract(ee.Image(neighbor_mean.select(['VV_mean', 'VH_mean'])))

    # Add covariates as bands
    img = img.addBands(spatial_mean.select(['VV', 'VH'])
                       .rename(['VV_mean', 'VH_mean']))\
        .addBands(ee.Image(neighbor_mean.select(['VV_mean', 'VH_mean']))
                  .rename(['VV_Nmean', 'VH_Nmean']))\
        .addBands(spatial_diff.select(['VV', 'VH'])
                  .rename(['VV_diff', 'VH_diff']))\
        .addBands(neighbor_diff.select(['VV', 'VH'])
                  .rename(['VV_Ndiff', 'VH_Ndiff']))

    """Step 4: Spatially smooth the input independant bands by applying a
    median smoother to each band
    with a circle kernel of 1.5 pixel radius
    """
    img = img.select(indep_variables).focal_median().toFloat()
    return img


def pair_opt_SAR(optical_collection, SAR_collection, AOI, indep_variables):
    """
    Step 5 Pair optical and SAR image collections
    """
    def pair_image(img):
        """ Step5: Pair Optical and SAR images according to collection date
        SAR images within 12 days before and 12 days after the optical images
        are composited with the mean of image stack at each pixel"""

        S2_date = ee.Date(img.get('system:time_start'))

        # Filtered S1 images within the 25 days time window centered at S2
        # collection date
        S1_filtered = SAR_collection.filterDate(
            S2_date.advance(-12, 'day'), S2_date.advance(12, 'day'))
        S1_composite = S1_filtered.mean()

        # Calculate covariates
        S1_covariates = cal_covariates(S1_composite, AOI, indep_variables)
        img = img.set({'S1_COUNT': S1_filtered.size()})
        img = ee.Algorithms.If(
            S1_filtered.size().gt(0),
            img.addBands(S1_covariates),
            img)
        return ee.Image(img)

    opt_SAR = optical_collection.map(pair_image).filterMetadata(
        'S1_COUNT', 'greater_than', 0)
    return opt_SAR


def Temporal_PCA(imageCollection, AOI, numComponent=10, scale=10):
    """
    Apply pixelwise PCA analysis along each time series
    """
    # Convert a single band imagecollection to a multiband image
    image = imageCollection.toBands().clip(AOI)
    band_names = image.bandNames()
    # Mean center the data to enable a faster covariance reducer
    # and an standard stretch of the principal components.
    mean_dict = image.reduceRegion(
        ee.Reducer.mean(), AOI, scale, maxPixels=1e9)
    means = ee.Image.constant(mean_dict.values(band_names))
    centered = image.subtract(means)

    # Collapse the bands of the image into a 1D array per pixel.
    arrays = centered.toArray()

    # Compute the covariance of the bands within the region.
    covar = arrays.reduceRegion(
        ee.Reducer.centeredCovariance(),
        AOI,
        scale,
        maxPixels=1e9)

    # Get the 'array' covariance result and cast to an array.
    # This represents the band-to-band covariance within the region.
    covar_array = ee.Array(covar.get('array'))

    # Perform an eigen analysis and slice apart the values and vectors.
    eigens = covar_array.eigen()

    # This is a P-length vector of eigen_values.
    eigen_values = eigens.slice(1, 0, 1)
    # This is a PxP matrix with eigen_vectors in rows.
    eigen_vectors = eigens.slice(1, 1)

    # Convert the array image to 2D arrays for matrix computations.
    array_image = arrays.toArray(1)

    # Left multiply the image array by the matrix of eigen_vectors.
    principal_components = ee.Image(eigen_vectors).matrixMultiply(array_image)

    # Slice the eigen vector and pricipapComponents according to input
    # threshold
    eigen_slice = eigen_vectors.slice(0, 0, numComponent)
    pc_slice = principal_components.arraySlice(0, 0, numComponent)

    # Perform inverse PCA analysis with filtered components
    pc_inverse = ee.Image(eigen_slice.transpose()).matrixMultiply(pc_slice)

    # Added the means back
    smooth_results = pc_inverse.arrayProject(
        [0]).arrayFlatten(
        [band_names]).add(means)
    return smooth_results


def post_process(paired_collection, prediction,
                 AOI, std_cloud_threshold):
    """
    Calibrate NDVI predictions and fill gaps
    """

    def pred_standardize(img):
        """ Step9: Calibrate NDVI predictions by
        standardizing prediction according to the observation
        at the same time"""

        # Get the id of opt-SAR pair
        # Find the corresponding smoothed image from the multibands
        # NDVI_smoothed
        img_id = img.id().cat('_NDVI_pred')
        mask = img.select('Mask')
        cloud_cover = ee.Number(img.get('CLOUD_PERCENTAGE_AOI'))
        NDVI_pred = prediction.select(img_id).rename('NDVI')
        NDVI_obs = img.select('NDVI')

        # Calcuate the spatial mean and std of predictions and observations for
        # clear pixels
        reducer = ee.Reducer.mean().combine(
            ee.Reducer.intervalMean(
                85, 95), '90', True) .combine(
            ee.Reducer.intervalMean(
                5, 15), '10', True)

        pred_stats = NDVI_pred.updateMask(
            mask.eq(0)).reduceRegion(
            reducer, AOI, 100, maxPixels=1e9)
        obs_stats = NDVI_obs.updateMask(
            mask.eq(0)).reduceRegion(
            reducer, AOI, 100, maxPixels=1e9)

        pred_90mean = ee.Number(pred_stats.get('NDVI_90mean'))
        pred_10mean = ee.Number(pred_stats.get('NDVI_10mean'))
        pred_mean = ee.Number(pred_stats.get('NDVI_mean'))
        pred_uprange = pred_90mean.subtract(pred_mean)
        pred_downrange = pred_10mean.subtract(pred_mean)

        obs_90mean = ee.Number(obs_stats.get('NDVI_90mean'))
        obs_10mean = ee.Number(obs_stats.get('NDVI_10mean'))
        obs_mean = ee.Number(obs_stats.get('NDVI_mean'))
        obs_uprange = obs_90mean.subtract(obs_mean)
        obs_downrange = obs_10mean.subtract(obs_mean)

        pred_uprange = ee.Number(
            ee.Algorithms.If(pred_uprange.neq(0), pred_uprange, obs_uprange))
        pred_downrange = ee.Number(
            ee.Algorithms.If(
                pred_downrange.neq(0),
                pred_downrange,
                obs_downrange))

        # Normalize predictions according to predictions at the same collection
        # time
        NDVI_diff = NDVI_pred.subtract(pred_mean)
        NDVI_upcalibrated = NDVI_diff.divide(pred_uprange)\
            .multiply(obs_uprange).add(obs_mean).rename('NDVI_pred')
        NDVI_downcalibrated = NDVI_diff.divide(pred_downrange)\
            .multiply(obs_downrange).add(obs_mean).rename('NDVI_pred')

        NDVI_calibrated = NDVI_upcalibrated.where(
            NDVI_diff.lt(0), NDVI_downcalibrated)
        NDVI_calibrated = ee.Algorithms.If(
            cloud_cover.lt(std_cloud_threshold),
            NDVI_calibrated,
            NDVI_pred)
        return img.addBands(
            ee.Image(NDVI_calibrated).rename('NDVI_pred'),
            overwrite=True)

    NDVI_calibrated = paired_collection.map(pred_standardize)

    def gap_infilling(img):
        """Step10: Replace cloudy NDVI with predictions"""
        prediction = img.select('NDVI_pred')
        observation = img.select('NDVI')
        cloudMask = img.select('Mask')

        # Relpace observation with prediction where cloud is identified
        gap_filled = observation.where(cloudMask, prediction).focal_median()

        # Calculate Mean absolute error (MAE) as the skill of prediction
        # -1 represents invalid MAE for image fully covered by cloud
        error = observation.subtract(
            prediction).abs().updateMask(cloudMask.eq(0))
        MAE = ee.Number(
            error.reduceRegion(
                ee.Reducer.mean(),
                AOI,
                100,
                maxPixels=1e9).get('NDVI'))
        MAE = ee.Algorithms.If(MAE, MAE, -1)

        # Save the final output and the cloud mask
        return ee.Image(gap_filled).addBands(cloudMask).toFloat().copyProperties(
            img,
            ['random', 'Split_label', 'CLOUD_PERCENTAGE_AOI',
             'system:time_start', 'system:index']
        ).set(
            {'MAE': MAE})

    NDVI_filled = NDVI_calibrated.map(gap_infilling)

    return NDVI_calibrated, NDVI_filled
