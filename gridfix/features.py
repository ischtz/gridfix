#!/usr/bin/python3
# -*- coding: utf-8 -*-

import types

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

from pandas import DataFrame, read_table
from scipy.ndimage import center_of_mass    # for CentralBiasFeature
from scipy.ndimage.filters import sobel     # for SobelEdgeFeature

from .regionset import RegionSet
from .model import ImageSet


class Feature(object):
    """ Base class for image features defined for each region in a set

    A Feature can be described as a two-step "recipe" on a given ImageSet and 
    RegionSet: First, a transform() is executed upon each image array, e.g. a
    filter operation. Second, a combine() operation yields a single value per region.
    The result is combined into a feature vector of len(regionset).

    Attributes:
        regionset (RegionSet): the associated regionset object
        imageset (ImageSet): a set of images or feature maps to process
        length (int): length of feature vector, i.e. number of regions
    """

    def __init__(self, regionset, imageset, trans_fun=None, comb_fun=None, label=None, normalize_output=False):
        """ Create a new basic Feature object.

        Args:
            regionset (RegionSet): a RegionSet object
            trans_fun (function): Function to use for the transformation step instead
                of default. Must accept PIL.Image and return Image or np.ndarray of same size.
            comb_fun (function): Function to use for the reduction step instead of default.
                Must accept a np.ndarray and return a scalar value.
            label (string): optional label to distinguish between Features of the same type
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """
        self.regionset = regionset
        self.imageset = imageset
        self.label = label
        self.normalize_output = normalize_output

        self._fvalues = {}

        # Replace default functions only if the specified ones make sense
        try:
            tmp = trans_fun(self, np.array([[0,1],[0,0]]))
            self._transform = self.transform   # keep default around
            self.transform = types.MethodType(trans_fun, self)
        except TypeError:
            if trans_fun is not None:
                print('Warning: trans_fun seems to return invalid values, using default!')

        try:
            tmp = comb_fun(self, np.array([[0,1],[0,0]]), np.array([[0,1],[0,0]]))
            self._combine = self.combine
            self.combine = types.MethodType(comb_fun, self)
        except TypeError:
            if comb_fun is not None:
                print('Warning: comb_fun seems to return invalid values, using default!')


    def __repr__(self):
        """ String representation for print summary. """
        desc = ''
        if self.label is not None:
            desc = ' "{:s}"'.format(str(self.label))

        norm = ''
        if self.normalize_output:
            norm = ', normalized'
        r = '<gridfix.{:s}{:s}, length={:d}{:s}>'.format(self.__class__.__name__, desc, len(self.regionset), norm)
        r += '\nRegions:\n\t{:s}'.format(str(self.regionset))
        r += '\nImages:\n\t{:s}'.format(str(self.imageset))

        return(r)


    def __len__(self):
        """ Overload len() to report length of feature vector. """
        return len(self.regionset)


    def __getitem__(self, imageid):
        """ Bracket indexing using an imageid returns the feature vector. """
        return self.apply(imageid, normalize=False)


    def transform(self, image):
        """ Apply image transform to specified image. 

        In the base Feature class, this simply returns the input image as-is.
        This function may be overloaded in subclasses or replaced by the 
        trans_fun= argument on construction time. 

        Args:
            image (ndarray): image / feature map array to transform

        Returns: 
            np.ndarray of image data
        """
        return np.asarray(image, dtype=float)


    def combine(self, image, region, fun=np.mean):
        """ Combine all selected pixel values in selection using specified function.

        In the base Feature class, this simply returns the mean of all pixels.
        This function may be overloaded in subclasses or replaced by the comb_fun= argument.

        Args:
            image (np.ndarray): 2D feature image
            region (np.ndarray): binary mask array defining a region
            fun (function): function to apply to selection. Must return a scalar.

        Returns:
            Scalar value depending on the specified function.
        """
        return fun(np.asarray(image[region], dtype=float))


    def apply(self, imageid, normalize=None):
        """ Apply feature to a single image from associated ImageSet.

        Args:
            imageid (str): valid ID from associated ImageSet
            normalize (bool): if True, scale output to range 0...1 (default: False)
        Returns:
            1D numpy.ndarray of feature values, same length as regionset
        """
        if imageid not in self._fvalues.keys():
            fv = []
            img_in = self.imageset[imageid]

            # Transformation step
            img_trans = self.transform(img_in)

            # Combination step
            for region in self.regionset[imageid]:
                f = self.combine(np.asarray(img_trans),  # make sure combine gets arrays
                                 np.asarray(region, dtype=bool))
                fv.append(f)

            # cache for later use
            self._fvalues[imageid] = np.array(fv)


        f = self._fvalues[imageid]
        if self.normalize_output:
            # Feature set to always normalize
            return((f - f.min()) / (f.max() - f.min()))
        elif normalize is not None:
            if normalize:
                # Upstream Model requested normalized values
                return((f - f.min()) / (f.max() - f.min()))
        else:
            # Default is to return unmodified values
            return f


    def apply_all(self, normalize=None):
        """ Apply feature to every image in the ImageSet and return a DataFrame.

        Args:
            normalize (bool): if True, scale output to range 0...1 (default: False)

        Returns: DataFrame similar to RegionSet.info with region feature values
        """
        rmeta = self.regionset.info.copy()
        if self.regionset.is_global:
            # If global RegionSet, copy region data once per image
            df = DataFrame(columns=rmeta.columns)
            for img in self.imageset.imageids:
                tmp = rmeta.copy()
                tmp['imageid'] = img
                df = df.append(tmp)
        else:
            df = rmeta
        df['value'] = np.nan

        for img in self.imageset.imageids:
            if img not in self.regionset.imageids and not self.regionset.is_global:
                continue

            img_trans = self.transform(self.imageset[img])
            l = self.regionset._select_labels(img)

            for idx,region in enumerate(self.regionset[img]):
                f = self.combine(np.asarray(img_trans), np.asarray(region, dtype=bool))
                df.loc[(df['imageid'] == img) & (df['region'] == str(l[idx])), 'value'] = f

        # Normalize value column if requested
        if self.normalize_output:
            val = df.loc[:, 'value']
            df.loc[:, 'value'] = (val - val.min()) / (val.max() - val.min())

        elif normalize is not None:
            if normalize:
                val = df.loc[:, 'value']
                df.loc[:, 'value'] = (val - val.min()) / (val.max() - val.min())

        return df


    def plot(self, imageid, what='both', cmap='gray', image_only=False, ax=None):
        """ Display feature map and/or feature values.

        Args:
            imageid (str): valid ID from associated ImageSet
            what (str): show only feature 'values', 'map' or 'both'
            cmap (str): name of matplotlib colormap to use
            image_only (boolean): if True, return only image content without labels
            ax (Axes): axes object to draw to (only if what!='both')

        Returns:
            matplotlib figure object, or None if passed an axis to draw on
        """
        lbl_text = ''
        if self.label is not None:
            lbl_text = ': {:s}'.format(str(self.label))

        if what == 'map':
            if ax is not None:
                ax1 = ax
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,1,1)

            imap = self.transform(self.imageset[imageid])
            ax1.imshow(imap, cmap=plt.get_cmap(cmap), interpolation='none')
            if image_only:
                ax1.axis('off')
            else:
                ax1.set_title('{:s}{:s} (img: {:s})'.format(self.__class__.__name__, lbl_text, imageid))

        elif what == 'values':
            if ax is not None:
                ax1 = ax
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,1,1)

            self.regionset.plot(values=self.apply(imageid), imageid=imageid, cmap=cmap, ax=ax1, image_only=image_only)
            if image_only:
                ax1.axis('off')
            else:
                ax1.set_title('{:s}{:s} (img: {:s})'.format(self.__class__.__name__, lbl_text, imageid))

        else:
            if ax is not None:
                print('Warning: ax argument is only valid if a single plot type is specified! Returning full figure instead.')

            fig = plt.figure()
            ax1 = fig.add_subplot(1,2,1)
            imap = self.transform(self.imageset[imageid])
            ax1.imshow(imap, cmap=plt.get_cmap(cmap), interpolation='none')
            ax2 = fig.add_subplot(1,2,2)
            self.regionset.plot(values=self.apply(imageid), imageid=imageid, cmap=cmap, ax=ax2, image_only=image_only)
            if image_only:
                ax1.axis('off')
                ax2.axis('off')
            else:
                ax1.set_title('map')
                ax2.set_title('values')
                fig.suptitle('{:s}{:s} (img: {:s})'.format(self.__class__.__name__, lbl_text, imageid))

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig



class CentralBiasFeature(Feature):
    """ Models central viewer bias as the distance to image center for each region. 

    The exact model of distance depends on the "measure" argument: if set to 'distance'
    (the default), the image transformation step does nothing and euclidean distance is 
    returned. If set to 'gaussian', anisotropic Gaussian distance based on Clarke & Tatler, 
    2014, Vis Res is used and transform() returns the corresponding Gaussian map. 
    """

    def __init__(self, regionset, imageset, measure='gaussian', sig2=0.23, nu=None, label=None, normalize_output=False):
        """ Create a new CentralBiasFeature object.

        Args:
            regionset (RegionSet): region set to apply feature to
            imageset (ImageSet): a set of images or feature maps (optional for this Feature)
            measure (string): distance measure to use ('euclidean', 'gaussian', 'taxicab')
            sig2 (float): variance value for type='gaussian'
            nu (float): anisotropy value for type='gaussian'
            label (string): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """
        if measure not in ['gaussian', 'euclidean', 'taxicab']:
            print('Warning: unknown central bias measure "{:s}" specified, falling back to euclidean distance!'.format(measure))
            measure = 'euclidean'

        self.measure = measure
        self._map = None
        self._values = {}

        self.sig2 = sig2
        self.nu = nu

        if nu is None:
            if measure == 'gaussian':
                self.nu = 0.45
            else:
                self.nu = 1
            
        
        def _transform(self, image=None):
            """ For 'gaussian': Gauss distance map, else return empty array. """

            if self._map is None:
                # compute feature map on first call

                mapsize = self.imageset.size[1], self.imageset.size[0]

                if measure == 'gaussian':
                    self._map = self._aniso_gauss(mapsize, self.sig2, self.nu)
                else:
                    self._map = np.zeros(mapsize, dtype=float)

            return self._map
            

        def _combine(self, image, region):
            """ Calculate distance from region center of mass to image center. """
            
            mapsize = self.imageset.size

            com = np.round(center_of_mass(region))
            img_center = (int(round(mapsize[1] / 2)), int(round(mapsize[0] / 2)))

            if self.measure == 'gaussian':
                return image[int(com[0]), int(com[1])]
        
            elif self.measure == 'euclidean':
                dist = np.hypot((com[0] - img_center[0]) / self.nu, com[1] - img_center[1])
                return np.round(dist, 0)

            elif measure == 'taxicab':
                dist = abs(com[0] - img_center[0]) + abs(com[1] - img_center[1])
                return np.round(dist, 0)

        self._trans_fun = _transform
        self._comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)


    def apply(self, imageid=None, normalize=None):
        """ Apply central bias to image, returning region distance values.

        Args:
            imageid (str): for consistency, ignored for central bias (same for all images)
            normalize (boolean): if True, scale output to range 0...1 (default: False)

        Returns:
            1D numpy.ndarray of feature values, same length as regionset
        """
        if imageid not in self._values.keys():
            fv = []
            
            img_trans = self.transform()

            for region in self.regionset[imageid]:
                f = self.combine(img_trans, np.asarray(region, dtype=bool))
                fv.append(f)

            self._values[imageid] = np.array(fv)

        f = self._values[imageid]
        if self.normalize_output:
            # Feature set to always normalize
            return((f - f.min()) / (f.max() - f.min()))
        elif normalize is not None:
            if normalize:
                # Upstream Model requested normalized values
                return((f - f.min()) / (f.max() - f.min()))
        else:
            # Default is to return unmodified values
            return f


    def __repr__(self):
        """ String representation for printing """
        desc = ''
        if self.label is not None:
            desc = ' "{:s}"'.format(str(self.label))

        r = '<gridfix.CentralBiasFeature{:s}, length={:d}, measure "{:s}"'.format(desc, len(self.regionset), self.measure)

        if self.measure == 'gaussian':
            r += ', sig2={:.2f}, nu={:.2f}'.format(self.sig2, self.nu)

        r += '>'
        r += '\nRegions:\n\t{:s}'.format(str(self.regionset))
        r += '\nImages:\n\t{:s}'.format(str(self.imageset))
        return(r)


    def _aniso_gauss(self, shape, sig2=0.23, nu=0.45):
        """ Calculate anisotropic, aspect-corrected Gaussian central bias map.

        This function yields an image-sized Gaussian map of the distance
        from image center, including anisotropy as described in Clarke &
        Tatler, 2014, Vis Res.

        Args:
            shape (tuple): image ndarray shape (height, width)
            sig2 (float): variance of Gaussian
            nu (float): anisotropy coefficient

        Returns:
            2D ndarray containing Gaussian central distance map
            (values increase with larger distance from center)
        """
        height, width = shape[0], shape[1]
        ar = width / height
        x = np.linspace(-1, 1, width)
        y = np.linspace(-1 / ar , 1 / ar, height)
        (xx, yy) = np.meshgrid(x, y)

        G = np.exp(-((xx ** 2) / sig2 + (yy ** 2) / sig2 / nu) / 2)
        return 1 - G



class LuminanceFeature(Feature):
    """ Models mean image luminance in each region """

    def __init__(self, regionset, imageset, label=None, normalize_output=False):
        """ Create a new LuminanceFeature
        
        Args:
            regionset: a RegionSet to be evaluated
            imageset: ImageSet containing images or feature maps to process
            label (str): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1

        """

        def _transform(self, image):
            """ Convert 3D-RGB image to 2D-intensity (like rgb2gray.m). """
            if image.ndim > 2 and image.shape[2] == 3:
                R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                lum = 0.2989 * R + 0.5870 * G + 0.1140 * B
                return lum
            else:
                return image

        def _combine(self, image, region):
            """ Return mean region luminance. """
            return(image[region].mean())

        self.trans_fun = _transform
        self.comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)



class LumContrastFeature(Feature):
    """ Feature based on local luminance contrast in each region """

    def __init__(self, regionset, imageset, label=None, normalize_output=False):
        """ Create a new LuminanceContrastFeature

        Args:
            regionset: a RegionSet to be evaluated
            imageset: ImageSet containing images or feature maps to process
            label (str): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """

        def _transform(self, image):
            """ Convert 3D-RGB image to 2D-intensity (like rgb2gray.m). """
            if image.ndim > 2 and image.shape[2] == 3:
                R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                lum = 0.2989 * R + 0.5870 * G + 0.1140 * B
                return lum
            else:
                return image

        def _combine(self, image, region):
            """ Return local contrast of luminance image. """
            return(image[region].std() / image.mean())

        self.trans_fun = _transform
        self.comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)



class SobelEdgeFeature(Feature):
    """ Feature based on relative prevalence of edges within each region """

    def __init__(self, regionset, imageset, label=None, normalize_output=False):
        """ Create a new SobelEdgeFeature

        Args:
            regionset: a RegionSet to be evaluated
            imageset: ImageSet containing images or feature maps to process
            label (str): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """

        def _transform(self, image):
            """ Run sobel filter on luminance image, then binarize. """
            if image.ndim > 2 and image.shape[2] == 3:
                R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                lum = 0.2989 * R + 0.5870 * G + 0.1140 * B
            else:
                lum = image

            sx = sobel(lum, 0)
            sy = sobel(lum, 1)
            si = np.hypot(sx, sy)

            # Determine threshold for edge detection (adapted from MATLAB edge.m)
            scale = 4.0
            cutoff = scale * si.mean()
            thresh = np.sqrt(cutoff)

            return np.asarray(si >= thresh, dtype=np.uint8)


        def _combine(self, image, region):
            """ Mean of binary image yields fraction of edges. """
            return(image[region].mean())

        self.trans_fun = _transform
        self.comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)



class MapFeature(Feature):
    """ Feature to apply a statistical function to each region in feature maps

    Attributes:
        stat (function): the statistics function to apply to each region
    """

    def __init__(self, regionset, imageset, stat=np.mean, label=None, normalize_output=False):
        """ Create a new MapFeature

        Args:
            regionset: a RegionSet to be evaluated
            imageset: ImageSet containing images or feature maps to process
            stat (function): the statistics function to apply to each region
            label (str): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """
        self.stat = stat

        def _transform(self, image):
            """ Does nothing, expects predefined input map! """
            return image

        def _combine(self, image, region):
            """ Applies stat function to region. """
            return(self.stat(image[region]))

        self.trans_fun = _transform
        self.comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)



class RegionFeature(Feature):
    """ Feature based on properties of the regions in a RegionSet. Allows to use
        e.g. region area or fraction of image pixels as a model predictor.

    Attributes:
        region_property (str): column from DataFrame to return as region feature
    """

    def __init__(self, regionset, imageset, region_property='area', label=None, normalize_output=False):
        """ Create a new RegionFeature

        Args:
            regionset: a RegionSet to be evaluated
            imageset: ImageSet containing images or feature maps to process
            region_property (str): column from DataFrame to return as region feature
            label (str): optional label to distinguish between Features
            normalize_output (bool): if True, always normalize output values of this feature to 0..1
        """
        if region_property not in regionset.info.columns:
            raise ValueError('Specified region property is not a column in RegionSet.info DataFrame! Example: "area"')
        self.region_property = region_property

        def _transform(self, image):
            """ Does nothing """
            return image

        def _combine(self, image, region):
            """ Does nothing """
            return None

        self.trans_fun = _transform
        self.comb_fun = _combine

        Feature.__init__(self, regionset, imageset, trans_fun=_transform, comb_fun=_combine, label=label,
                         normalize_output=normalize_output)


    def apply(self, imageid, normalize=None):
        """ Return selected region property of each region for specified imageid.

        Args:
            imageid (str): valid ID from associated ImageSet
            normalize (bool): if True, scale output to range 0...1 (default: False)

        Returns:
            1D numpy.ndarray of feature values, same length as regionset
        """
        if self.regionset.is_global:
            imageid = '*'
        if imageid not in self.regionset._regions.keys():
            raise ValueError('The imageid specified for RegionFeature was not found in the associated Imageset!')

        sel_df = self.regionset.info[self.regionset.info.imageid == imageid]

        f = np.array(sel_df[self.region_property])
        if self.normalize_output:
            # Feature set to always normalize
            return((f - f.min()) / (f.max() - f.min()))
        elif normalize is not None:
            if normalize:
                # Upstream Model requested normalized values
                return((f - f.min()) / (f.max() - f.min()))
        else:
            # Default is to return unmodified values
            return f



