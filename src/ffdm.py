from __future__ import absolute_import, division, print_function
import pydicom
import numpy as np
import os
from skimage.transform import rescale


class FFDM:
    def __init__(self, dicom_file: str):
        self.__path = dicom_file
        self.id = ''
        self.source = 'Unknown'
        self.israw = False
        self.view = ''
        self.side = ''
        self.tomo = ''
        self.date = ''
        self.istomo = False
        self.ismlo = False
        self.iscc = False
        self.H = float('nan')
        self.KVP = float('nan')
        self.psize = float('nan')
        self.cforce = float('nan')
        self.target = 'NA'
        self.filter = 'NA'
        self.age = '055Y'

        assert os.path.exists(dicom_file), 'The file does not exist'
        ds = pydicom.dcmread(dicom_file)
        self.__img = ds.pixel_array

        # Is tomo
        if ('SeriesDescription' in ds) and (ds.SeriesDescription.find('Tomosynthesis') != -1):
            self.istomo = True
            if ds.SeriesDescription.find('Reconstruction'):
                self.tomo = 'RC'
            else:
                self.tomo = 'PR'
            self.israw = ds.SeriesDescription.find('Raw') != -1

        # Is raw
        if 'PresentationIntentType' in ds:
            if ds.PresentationIntentType.find('FOR PRESENTATION') == -1:
                self.israw = True

        # Accession number
        if 'AccessionNumber' in ds:
            self.id = ds.AccessionNumber

        # View
        if ('ViewPosition' in ds) and (ds.ViewPosition.find('MLO') != -1):
            self.view = 'MLO'
        elif ('ViewPosition' in ds) and (ds.ViewPosition.find('CC') != -1):
            self.view = 'CC'
        elif ('SeriesDescription' in ds) and (ds.SeriesDescription.find('MLO') != -1):
            self.view = 'MLO'
        elif ('SeriesDescription' in ds) and (ds.SeriesDescription.find('CC') != -1):
            self.view = 'CC'
        elif ('ProtocolName' in ds) and (ds.ProtocolName.find('MLO') != -1):
            self.view = 'MLO'
        elif ('ProtocolName' in ds) and (ds.ProtocolName.find('CC') != -1):
            self.view = 'CC'
        elif 'AcquisitionDeviceProcessingDescription' in ds:
            if ds.AcquisitionDeviceProcessingDescription.find('MLO') != -1:
                self.view = 'MLO'
            elif ds.AcquisitionDeviceProcessingDescription.find('CC') != -1:
                self.view = 'CC'
            else:
                self.view = ''
        elif ('ViewPosition' in ds) and (ds.ViewPosition != ''):
            self.view = ds.ViewPosition.replace(" ", "")

        self.ismlo = self.view == 'MLO'
        self.iscc = self.view == 'CC'

        # Side
        if 'Laterality' in ds and ds.Laterality:
            self.side = ds.Laterality
        elif 'ImageLaterality' in ds:
            self.side = ds.ImageLaterality

        # Manufacturer
        if 'Manufacturer' in ds:
            self.source = ds.Manufacturer

        # Body thickness and KVP
        if 'BodyPartThickness' in ds:
            self.H = ds.BodyPartThickness
        if 'KVP' in ds:
            self.KVP = ds.KVP

        # Spatial resolution
        if 'SpatialResolution' in ds:
            self.psize = ds.SpatialResolution
        elif 'DetectorElementPhysicalSize' in ds:
            self.psize = float(ds.DetectorElementPhysicalSize[0])
        elif 'DetectorElementSpacing' in ds:
            self.psize = float(ds.DetectorElementSpacing[0])
        elif 'PixelSpacing' in ds:
            self.psize = float(ds.PixelSpacing[0])

        # Target material
        if 'AnodeTargetMaterial' in ds and ds.AnodeTargetMaterial:
            self.target = ds.AnodeTargetMaterial[0:2]

        # Filter material
        if 'FilterMaterial' in ds and ds.FilterMaterial:
            self.filter = ds.FilterMaterial[0:2]

        # Date
        if 'StudyDate' in ds:
            self.date = ds.StudyDate

        # Compression force
        if 'CompressionForce' in ds and ds.CompressionForce:
            self.cforce = float(ds.CompressionForce)

        # Patient age
        if 'PatientAge' in ds and ds.PatientAge:
            self.age = ds.PatientAge

    def get_info(self):
        info = {}
        for a in dir(self):
            if not a.startswith('__') and not a.startswith('_FFDM__') and not callable(getattr(self, a)):
                info[a] = self.__getattribute__(a)
        return info

    def read(self, processed=True, res=0.1):
        im = self.__img
        # im = self.__img if self.side == 'L' else np.fliplr(self.__img)
        # im = np.asarray(self.__img, dtype=np.float64)

        # Find vendor
        vendor = self.source.upper()
        is_neg = (vendor.find('FUJIFILM') != -1) or \
                 (vendor.find('SECTRA') != -1) or  \
                 (vendor.find('PHILIPS') != -1)
        is_agfa = vendor.find('AGFA') != -1

        # normalize
        if self.israw:
            gmax = (2 ** 14) - 1
            gmin = 1
            im[im < gmin] = gmin
            im[im > gmax] = gmax
            imn = np.asarray(im, dtype=np.float32)
            imn = np.log(imn)
            imn = (np.amax(imn) - imn)**2
        elif is_neg:
            gmax = np.amax(im)
            gmin = 0
            im = gmax - im
            im[im < gmin] = gmin
            im[im > gmax] = gmax
            imn = np.asarray(im, dtype=np.float32)
            imn = (im - np.amin(imn)) / (np.amax(im) - np.amin(im))
        elif is_agfa:
            gmax = np.amax(im)
            gmin = 0
            im = gmax - im
            im[im < gmin] = gmin
            im[im > gmax] = gmax
            imn = np.asarray(im, dtype=np.float32)
            imn = (im - np.amin(imn)) / (np.amax(im) - np.amin(im))
        else:
            gmax = (2 ** 12) - 1
            gmin = 0
            im[im < gmin] = gmin
            im[im > gmax] = gmax
            imn = np.asarray(im, dtype=np.float32)
            imn = (imn - np.amin(imn)) / (np.amax(imn) - np.amin(imn))

        # Standard resolution
        if res is not None:
            if self.psize/res < 1:
                imn = rescale(imn, self.psize/res, anti_aliasing=True, multichannel=False)
                imn = np.asarray(imn, dtype=np.float32)

        # return imn, im_clahe, im
        if processed:
            return imn
        else:
            return im
