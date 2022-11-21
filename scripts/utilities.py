# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 15:02:10 2021

@author: uqymao1
"""
import ee
import time
ee.Initialize()

# Define expections when calling API


class Error(Exception):
    """Base class for other exceptions"""
    pass


class InvalidInputs(Error):
    """Raised when the inputs are invalid"""
    pass


class GEEError(Error):
    """Raised when the GEE implementation is problematic"""
    pass


"""Define functions """

# Define functions to export images


def check_task_status(task, cancel_when_interrupted=True):
    """This function checks the status of expoorting tasts every minute
    If the cancel_when_interrupted is set to True, the task will keep running
    on server while the python programme is interrupted.
    You can check the status on
    https://code.earthengine.google.com/
    """
    done = False
    while not done:
        try:
            task_status = task.status()
            if task_status['state'] == 'RUNNING':
                print(
                    "Exporting:{}, state:{}".format(
                        task_status['description'], task_status['state']
                    )
                )
                time.sleep(60)

            elif task_status['state'] == 'READY':
                print('Start exporting {}'.format(task_status['description']))
                time.sleep(60)

            elif task_status['state'] == 'FAILED':
                print(task_status['error_message'])
                raise GEEError

            elif task_status['state'] == 'COMPLETED':
                done = True
                print(
                    "{} have been uploaded to GEE asset successfully".format(
                        task_status['description']))
            else:
                print("Task has been canceled through GEE code Editor")
                raise GEEError
        except GEEError:
            raise
        except KeyboardInterrupt:
            if cancel_when_interrupted:
                task.cancel()
            print('Task is canceled')
            raise


def export_image_toasset(image, AOI, assetID, description=''):
    """This function exports images to GEE asset"""
    task = ee.batch.Export.image.toAsset(**{
        'image': image,
        'description': description,
        'assetId': assetID,
        'scale': 10,
        'region': AOI,
        'maxPixels': 1e9
    })
    task.start()
    return task


def export_image_todrive(image, AOI, filename, folder, description=''):
    """This function exports images to Google drive"""
    task = ee.batch.Export.image.toDrive(**{
        'image': image,
        'description': description,
        'fileNamePrefix': filename,
        'folder': folder,
        'scale': 10,
        'region': AOI
    })
    task.start()
    return task


def export_table_todrive(table, filename, folder, description=''):
    """This function exports featureCollection (table) to Google drive"""
    task = ee.batch.Export.table.toDrive(**{
        'collection': table,
        'description': description,
        'fileNamePrefix': filename,
        'folder': folder,
        'fileFormat': 'csv'
    })
    task.start()
    return task
