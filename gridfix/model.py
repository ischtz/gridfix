#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import types
import itertools

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt

from PIL import Image
from pandas import DataFrame, read_table

from scipy.io import whosmat, loadmat


class ImageSet(object):
    """ Set of images of equal size for masking and Feature creation. 

    Attributes:
        info (DataFrame): table of image metadata (filenames, size, type...)
        imageids (list): All unique image IDs in the set
        label (str): optional label to distinguish between ImageSets
        mat_var (str): name of MATLAB variable name if imported from .mat
        mat_contents (list): list of all variable names in .mat if applicable
        normalize (boolean): True if image data was normalized to 0..1 range
        preload (boolean): True if images were preloaded into memory
        size (tuple): image dimensions, specified as (width, height)
    """

    def __init__(self, images, mat_var=None, size=None, imageids=None, 
                 sep='\t', label=None, normalize=None, preload=False):
        """ Create a new ImageSet and add specified images 

        Args:
            images: one of the following:
                - path to a single image or .mat file
                - path to a folder containing image or .mat files
                - list of image or .mat file names
                - simple text file, one filename per line
                - text / CSV file containing columns 'filename' and 'imageid'
            mat_var (str): variable to use if _images_ is a MATLAB file
            size (tuple): image dimensions, specified as (width, height) in pixels
            imageids (list): string ID labels, one for each image. If not specified,
                file names without extension or a numerical label 0..n will be used
            sep (str): if _images_ is a text file, use this separator
            label (string): optional descriptive label
            normalize (boolean): normalize image color / luminance values to 0...1
                (defaults to True for images, False for MATLAB data)
            preload (boolean): if True, load all images at creation time
                (faster, but uses a lot of memory)
        """
        self.imageids = []
        self._images = {}

        # DF to hold image metadata
        df_col = ['imageid', 'filename', 'width', 'height', 'channels', 'type', 'mat_var']
        self.info = DataFrame(columns=df_col)

        self.size = None
        self.label = label
        self.normalize = normalize
        self.preload = preload
        self._last_image = None
        self._last_imageid = None

        self.mat_var = None
        self.mat_contents = None

        if size is not None:
            self.size = tuple(size)

        self._add_images(images, mat_var, imageids, sep)


    def __repr__(self):
        """ Short string representation for printing """
        # Image size
        if self.size is None:
            size = 'undefined'
        else:
            size = str(self.size)

        # Number of images
        s = ''
        if len(self.imageids) != 1:
            s = 's'

        desc = ''
        if self.label is not None:
            desc = ' "{:s}"'.format(str(self.label))

        mat = ''
        if self.mat_var is not None:
            mat = ', mat_var={:s}'.format(self.mat_var)
        norm = ''
        if self.normalize:
            norm = ', normalized'
        r = '<gridfix.ImageSet{:s}, {:d} image{:s}, size={:s}{:s}{:s}>'
        return r.format(desc, len(self.imageids), s, size, mat, norm)


    def __len__(self):
        """ Overload len(ImageSet) to report number of images. """
        return len(self.imageids)


    def __iter__(self):
        """ Overload iteration to step through the ndarray representations of images. """
        return iter([np.array(i) for i in self.images.keys()])


    def __getitem__(self, imageid):
        """ Allow to retrieve image by bracket indexing """
        return self.image(imageid)


    def _add_images(self, images, mat_var=None, imageids=None, sep='\t'):
        """ Add one or more image(s) to set. 

        Args:
            images: one of the following: 
                - path to a single image file
                - path to a folder containing image files
                - list of image file names
                - simple text file, one image filename per line
                - text / CSV file containing columns 'filename' and 'imageid'
            mat_var (str): variable to use if _images_ is a MATLAB file
            imageids (list): string ID labels, one for each image. If not specified,
                file names without extension or a numerical label 0..n will be used
            sep (str): if _images_ is a text file, use this separator
        """
        filelist = []
        if imageids is None:
            imageids = []

        img_root = os.getcwd()

        # Build file list
        if type(images) == list:
            filelist = images
        
        elif type(images) == str:

            if os.path.isdir(images):
                # Directory
                filelist = [os.path.join(images, b) for b in sorted(os.listdir(images))]

            elif os.path.isfile(images):
                # Single file - check if it is a text list of images!
                (ifname, ifext) = os.path.splitext(images)
                
                if ifext.lower() in ['.txt', '.tsv', '.csv', '.dat']:
                    # assume image list
                    try:
                        imgfiles = read_table(images, header=None, index_col=False, sep=sep)

                        if imgfiles.shape[1] == 1:
                            # Only one column: assume no headers and load as list of filenames
                            filelist = list(imgfiles.ix[:, 0])

                        elif imgfiles.shape[1] > 1:
                            # More than one column: look for 'imageid' and 'filename' columns
                            if 'imageid' in list(imgfiles.ix[0, :]) and 'filename' in list(imgfiles.ix[0, :]):
                                imgfiles.columns = imgfiles.ix[0, :]
                                imgfiles = imgfiles.ix[1:, :]
                                filelist = list(imgfiles['filename'])
                                imageids = list(imgfiles['imageid'])

                        lfolder, lname = os.path.split(images)
                        if len(lfolder) > 0:
                            img_root = os.path.abspath(lfolder)

                    except: 
                        raise ValueError('could not read image list file, check format!')
                else:
                    # assume single image file
                    filelist = [images]
        else:
            raise ValueError('first argument must be list or a string containing a file or directory!')

        # Verify image files
        filetable = []
        for (idx, ifile) in enumerate(filelist):
            try:
                imeta = self._verify_image(ifile, mat_var, img_root)

                # assign imageid
                (ifdir, iffile) = os.path.split(ifile)
                (ifbase, ifext) = os.path.splitext(iffile)
                if imageids is not None and len(imageids) > 0:
                    imageid = imageids.pop(0)
                else:
                    imageid = iffile
                imeta['imageid'] = imageid

                if imageid in self.imageids:
                    print('Warning: replacing existing image with ID <{:s}>!'.format(imageid))
                else:
                    self.imageids.append(imageid)

                filetable.append(imeta)

            except ValueError as err:
                print('Warning: file {:s} could not be added, error: {:s}!'.format(ifile, str(err)))

        df_col = ['imageid', 'filename', 'width', 'height', 'channels', 'type', 'mat_var']
        self.info = DataFrame(filetable, columns=df_col)
        self.imageids = list(self.info.imageid)

        # Preload files if requested
        for imid in self.imageids:
            if self.preload:
                self._images[imid] = self._load_image(imid)
            else:
                self._images[imid] = None


    def image(self, imageid):
        """ Get image by imageid, loading it if not preloaded

        Args:
            imageid (str): valid imageid from set

        Returns:
            ndarray of raw image data
        """
        if imageid not in self.imageids:
            raise ValueError('Specified _imageid_ does not exist!')

        if self._images[imageid] is not None:
            return self._images[imageid]
        else:
            if self._last_imageid == imageid:
                return self._last_image
            else:
                return self._load_image(imageid)


    def _load_image(self, imageid):
        """ Load and return image data for imageid. 

        Args:
            imageid (str): valid imageid from set

        Returns:
            ndarray of raw image data
        """
        if imageid not in self.imageids:
            raise ValueError('Specified _imageid_ does not exist!')

        if self._last_imageid is not None and self._last_imageid == imageid:
            return self._last_image

        imdata = None
        imeta = self.info[self.info.imageid == imageid]

        if len(imeta) > 0:

            if imeta.type.iloc[0] == 'MAT':
                mat = loadmat(imeta.filename.iloc[0], variable_names=imeta.mat_var.iloc[0])
                imdata = np.asarray(mat[imeta.mat_var.iloc[0]], dtype=float)

            else:
                i = Image.open(imeta.filename.iloc[0])
                if i.mode == 'RGBA':
                    i = i.convert('RGB')    # Drop alpha channel
                imdata = np.asarray(i, dtype=float)

            if self.normalize is None and imeta.type.iloc[0] != 'MAT':
                imdata = imdata / 255.0
                self.normalize = True
            elif self.normalize:
                imdata = imdata / 255.0

            self._last_imageid = imageid
            self._last_image = imdata

        return imdata


    def _verify_image(self, image_file, mat_var=None, img_root=None):
        """ Verify type and size of image without actually loading data

        Args: 
            image_file (str): path to image file to verify
            mat_var (str): optional variable name for MATLAB files
            img_root (str): folder containing image list in case file paths are relative

        Returns:
            dict of metadata, column names as in imageset.info DataFrame
        """
        if not os.path.isfile(image_file):
            if img_root is not None and os.path.isfile(os.path.join(img_root, image_file)):
                image_file = os.path.join(os.path.join(img_root, image_file))
            else:
                raise ValueError('file not found')

        (ifbase, ifext) = os.path.splitext(image_file)

        imeta = {'imageid': None, 
                 'filename': '',
                 'width': -1,
                 'height': -1, 
                 'channels': -1, 
                 'type': None,
                 'mat_var': None}
        
        # Matlab files
        if ifext.lower() == '.mat':
            try:
                # Load .mat header and identify variables
                tmp = whosmat(image_file)
                tmpvars = [m[0] for m in tmp]
                
                if self.mat_contents is None:
                    self.mat_contents = tmpvars
                
                if mat_var is None: 
                    mat_var = tmpvars[0]

                if mat_var in tmpvars:
                    
                    if self.mat_var is None:
                        self.mat_var = mat_var

                    # check image size
                    imshape = [m[1] for m in tmp if m[0] == mat_var][0]
                    if self.size is None:
                        self.size = (imshape[1], imshape[0])

                    if (imshape[1], imshape[0]) == self.size:
                        imeta['filename'] = image_file
                        imeta['width'] = imshape[1]
                        imeta['height'] = imshape[0]
                        if len(imshape) > 2:
                            imeta['channels'] = imshape[2]
                        else:
                            imeta['channels'] = 1
                        imeta['type'] = 'MAT'
                        imeta['mat_var'] = mat_var
                    
                    else:
                        w = 'Warning: skipping {:s} due to image size ({:d}x{:d} instead of {:d}x{:d}).'
                        print(w.format(iffile, imsize[0], imsize[1], self.size[0], self.size[1]))

                else:
                    raise ValueError('specified MATLAB variable not in file')

            except: 
                raise ValueError('error loading MATLAB data')

        # Image files
        else:
            try:
                i = Image.open(image_file) 
                imsize = i.size

                if self.size is None:
                    self.size = imsize

                # check image size
                if imsize == self.size:
                    imeta['filename'] = image_file
                    imeta['width'] = imsize[0]
                    imeta['height'] = imsize[1]
                    if i.mode in ['RGB', 'RGBA']:
                        imeta['channels'] = 3
                    else:
                        imeta['channels'] = 1
                    imeta['type'] = i.format
                    imeta['mat_var'] = ''
                    
                else:
                    w = 'Warning: skipping {:s} due to image size ({:d}x{:d} instead of {:d}x{:d}).'
                    print(w.format(image_file, imsize[0], imsize[1], self.size[0], self.size[1]))

            except OSError:
                raise ValueError('not an image or file could not be opened.')

        return imeta


    def plot(self, imageid, cmap='gray', image_only=False, ax=None):
        """ Display one of the contained images by imageid using matplotlib

        Args:
            imageid (str): valid imageid of image to show
            cmap (str): name of a matplotlib colormap to use
            image_only (boolean): if True, return only image content without labels
            ax (Axes): axes object to draw on, to include result in other figure

        Returns:
            matplotlib figure object, or None if passed an axis to draw on
        """
        try:
            if ax is not None:
                ax1 = ax
            else:
                fig = plt.figure()
                ax1 = fig.add_subplot(1,1,1)

            ax1.imshow(self.image(imageid), cmap=plt.get_cmap(cmap))
            if image_only:
                ax1.axis('off')
            else:
                ax1.set_title(imageid)

            if ax is None and not plt.isinteractive():
                # Only return figure object in non-interactive mode, otherwise
                # IPython/Jupyter will display the figure twice (once while plotting
                # and once as the cell result)!
                return fig

        except KeyError:
            raise ValueError('No image with ID "{:s}" in set'.format(imageid))


class Fixations(object):
    """ Dataset of fixation locations.

    Fixation locations are assumed to be one-indexed in input, e.g. 1..800 if 
    the image is 800 pixels wide, and converted internally to Pythons zero-indexed
    array convention.

    Attributes:
        data (DataFrame): DataFrame of raw fixation data 
        imageids (list): all unique image IDs represented in the dataset
        imageset (ImageSet): if present, the associated ImageSet
        input_file (str): file name of fixation data file
        num_samples (int): number of fixation samples
        num_vars (int): number of columns / variables in dataset 
        offset (tuple): optional offset from raw (x,y) positions
        shape (tuple): dimensions of .data, i.e. same as Fixations.data.shape
        variables (list): list of all variables loaded from input file
    """

    def __init__(self, data, imageset=None, offset=(0, 0), imageid='imageid', 
                 fixid='fixid', x='x', y='y', numericid=False):
        """ Create new Fixations dataset and calculate defaults.

        Args:
            data: a predefined DataFrame or name of a file containing fixation report
            imageset (ImageSet): if present, verify imageids against this ImageSet
            offset (tuple): an (x, y) tuple of pixel values to offset fixations. E.g.,
                if fixations have their origin at image center, use (-width/2, -height/2)
            imageid (str): name of data file column containing imageids
            fixid (str): name of data file column with unique fixation ID / index
            x (str): name of data file column for horizontal fixation locations
            y (str): name of data file column for vertical fixation locations
            numericid (boolean): if True, try to force parsing imageid as numeric
        """
        self.data = DataFrame()

        if isinstance(data, DataFrame):
            # data is already a DataFrame
            self.data = data
            self.input_file = None

        else:
            try:
                self.data = read_table(data, index_col=False)
                self.input_file = data

            except IOError:
                raise IOError('Could not load fixation data, check file name and type!')

        # Internal column names
        self._imageid = imageid
        self._fixid = fixid
        self._x = x
        self._y = y
        self._xpx = self._x + '_PX'
        self._ypx = self._y + '_PX'

        # Verify that all required columns are present
        cols = [imageid, fixid, x, y]
        miss_cols = []
        for c in cols:
            if c not in self.data.columns.values:
                miss_cols.append(c)

        if len(miss_cols) > 0:
            raise ValueError('Missing columns ({:s}), please specify column names!'.format(str(miss_cols)))
            
        # Image ID column should always be converted to strings
        if numericid:
            self.data[self._imageid] = self.data[self._imageid].astype(int).astype(str)
        else:
            self.data[self._imageid] = self.data[self._imageid].astype(str)

        self.imageids = list(self.data[self._imageid].unique())
        self.shape = self.data.shape
        self.num_samples = self.shape[0]
        self.num_vars = self.shape[1]
        self.variables = list(self.data.columns.values)

        # If ImageSet provided, check if all images are present
        self.imageset = None
        if imageset is not None:
            self.imageset = imageset
            missing_imgs = []
            for im in self.imageids:
                if im not in imageset.imageids:
                    missing_imgs.append(im)
            if len(missing_imgs) > 0:
                print('Warning: the following images appear in the fixation data but not the speficied ImageSet: {:s}'.format(', '.join(missing_imgs)))

        # Set offset and calculate pixel indices (_xpx, _ypx)
        self.offset = (0, 0)
        self.set_offset(offset)
        

    def __repr__(self):
        """ String representation """
        r = '<gridfix.Fixations data set, {:d} samples, {:d} images>'.format(self.num_samples, len(self.imageids))
        if self.imageset is not None:
            r += '\nImages:\n\t{:s}'.format(str(self.imageset))
        return r


    def __len__(self):
        """ Overload len() to report the number of samples """
        return self.data.shape[0]


    def __getitem__(self, imageid):
        """ Bracket indexing returns all fixations for a specified image """
        return self.select_fix(select={'imageid': imageid})


    def set_offset(self, offset):
        """ Set a constant offset for eye x/y coordinates.

        If image coordinates are relative to image center, use (-width/2, -height/2)
        (GridFix uses a coordinate origin at the top left).

        Args:
            offset (tuple): 2-tuple of (hor, ver) offset values in pixels
        """
        # Reset previous offset
        prevoffset = self.offset
        if prevoffset[0] != 0.0 or prevoffset[1] != 0.0:
            self.data[self._x] = self.data[self._x] - prevoffset[0]
            self.data[self._y] = self.data[self._y] - prevoffset[1]

        self.data[self._x] = self.data[self._x] + offset[0]
        self.data[self._y] = self.data[self._y] + offset[1]
        self.offset = (offset[0], offset[1])

        # Round x/y fixation positions to integers (pixels) while keeping original data
        # Note: we're going to use these as indices and Python is 0-indexed, so subtract 1!
        self.data[self._xpx] = np.asarray(np.round(self.data[self._x]), dtype=int) - 1
        self.data[self._ypx] = np.asarray(np.round(self.data[self._y]), dtype=int) - 1


    def select_fix(self, select={}):
        """ Return a subset of fixation data for specified imageid.

        Args:.
            select (dict): dict of filter variables, as {column: value}

        Returns:
            New Fixations object containing selected fixations only
        """
        if self._imageid not in select.keys():
            print('Warning: no image ID in filter variables, selection will yield fixations from multiple images! Proceed at own risk.')

        if select[self._imageid] not in self.data[self._imageid].values:
            print('Warning: zero fixations selected for specified imageid ({:s})'.format(select[self._imageid]))

        selection = self.data
        if len(select) > 0:
            for col, target in select.items():
                if col not in selection.columns.values:
                    print('Warning: filter variable {:s} not found in Fixations dataset!'.format(col))
                else:
                    # Make sure dict value is list-like
                    if type(target) not in (tuple, list):
                        target = [target]
                    selection = selection[selection[col].isin(target)]

        result = Fixations(selection.copy(), imageid=self._imageid, fixid=self._fixid,
                           x=self._x, y=self._y, imageset=self.imageset)
        return result


    def plot(self, imageid=None, select={}, on_image=True, oob=False,
             plotformat='wo', image_only=False, ax=None):
        """ Plot fixations for selected imageid, either alone or on image

        Args:
            imageid (str): optional image ID to plot fixations for
            select (dict): dict of additional filter variables (see select_fix())
            image (bool): if True, superimpose fixations onto image (if ImageSet present)
            oob (bool): if True, include out-of-bounds fixations when plotting
            plotformat (str): format string for plt.pyplot.plot(), default: white circles
            image_only (boolean): if True, return only image content without labels
            ax (Axes): axes object to draw to, to include result in other figure

        Returns:
            matplotlib figure object, or None if passed an axis to draw on
        """
        if imageid is not None:
            if imageid not in select.keys():
                select[self._imageid] = imageid
            plotfix = self.select_fix(select)
        else:
            plotfix = self

        if ax is not None:
            ax1 = ax 
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)

        try:
            if on_image:
                ax1.imshow(self.imageset.image(imageid), origin='upper')

        except AttributeError:
            print('Warning: cannot view fixations on image due to missing ImageSet!')

        if oob:
            ax1.plot(plotfix.data[self._xpx], plotfix.data[self._ypx], plotformat)

        else:
            try:
                if self.imageset is not None:
                    size = self.imageset.size
                else:
                    size = (max(plotfix.data[self._xpx]), max(plotfix.data[self._ypx]))
                fix = plotfix.data[(plotfix.data[self._xpx] >= 0) &
                                   (plotfix.data[self._xpx] < size[0]) &
                                   (plotfix.data[self._ypx] >= 0) &
                                   (plotfix.data[self._ypx] < size[1])]
                ax1.plot(fix[self._xpx], fix[self._ypx], plotformat)
                ax1.set_xlim((0, size[0]))
                ax1.set_ylim((0,size[1]))
                ax1.invert_yaxis()
                if image_only:
                    ax1.axis('off')
                else:
                    ax1.set_title(imageid)

            except AttributeError:
                print('Warning: cannot filter fixations for image boundaries due to missing ImageSet!')
                ax1.plot(plotfix[self._xpx], plotfix[self._ypx], plotformat)
                ax1.invert_yaxis()

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig


    def location_map(self, imageid=None, size=None):
        """ Binary ndarray of fixated and non-fixated pixels within image area

        Args:
            imageid (str): optional image ID to create map for one image only
            size (tuple): image dimensions, specified as (width, height).

        Returns:
            2d boolean ndarray, True where fixated, otherwise False
        """
        if imageid is not None:
            mapfix = self.select_fix({self._imageid: imageid})
        else:
            mapfix = self

        if size is None:
            if self.imageset is not None:
                size = self.imageset.size
            else:
                raise ValueError('Image size or attached ImageSet are necessary for location mapping!')

        fixloc = np.zeros((size[1], size[0]), dtype=bool)
        fix = mapfix.data[(self.data[self._xpx] >= 0) & (self.data[self._xpx] < size[0]) &
                          (self.data[self._ypx] >= 0) & (self.data[self._ypx] < size[1])]
        fixloc[fix[self._ypx], fix[self._xpx]] = True
        return fixloc


    def count_map(self, imageid=None, size=None):
        """ Map of fixation counts for each image pixel

        Args:
            imageid (str): optional image ID to create map for one image only
            size (tuple): image dimensions, specified as (width, height).

        Returns:
            2d ndarray of pixel fixation counts
        """
        if imageid is not None:
            mapfix = self.select_fix({self._imageid: imageid})
        else:
            mapfix = self

        if size is None:
            if self.imageset is not None:
                size = self.imageset.size
            else:
                raise ValueError('Image size or attached ImageSet are necessary for location mapping!')

        fixcount = np.zeros((size[1], size[0]), dtype=int)
        fix = mapfix.data[(self.data[self._xpx] >= 0) & (self.data[self._xpx] < size[0]) &
                          (self.data[self._ypx] >= 0) & (self.data[self._ypx] < size[1])]
        fixloc = fix[[self._ypx, self._xpx]].as_matrix()
        for pos in fixloc:
            fixcount[pos[0], pos[1]] += 1
        return fixcount



class FixationModel(object):
    """ Combines Features and Fixations to create predictors and R source for GLMM

    Attributes:
        chunks (list): list of data columns that define chunks (e.g., subjects or sessions)
        comp_features (dict): dict of labelled feature comparisons. Predictors will be replicated
            for each feature in a comparison so that features can serve as fixed or random factor
        exclude_first_fix (boolean): if True, first fixation index was set NaN (e.g.,, fixation cross)
        features (dict): dictionary of feature objects and feature groups
        predictors (DataFrame): model predictors for GLMM
        regionset (RegionSet): attached RegionSet
        runtime (float): time in seconds for most recent update of predictor matrix
    """

    def __init__(self, fixations, regionset, dv_type='fixated', features=None, feature_labels=None,
                 chunks=[], progress=False, exclude_first_fix=False):
        """ Create a new FixationModel.

        Args:
            fixations (Fixations): fixation data to use as model DV (column 'fixation')
            regionset (RegionSet): a RegionSet object defining length of all features
            dv_type (str): type of DV to generate:
                'fixated': binary coding of fixated (1) and unfixated (0) regions
                'count': absolute fixation count for each region
            features (list): list of Feature objects to add (use add_comparison for feature groups)
            feature_labels (list): string labels to apply to features defined using features= attribute
            chunks (list): list of fixation data columns that define chunks (e.g., subjects or sessions)
            progress (bool): print current image and group variables to indicate model build progress
            exclude_first_fix (bool): if True, set first fixated region per image to NaN for GLMM
        """
        self.regionset = regionset

        self._fix = fixations
        self._pred = DataFrame()
        self._consistent = False    # Flags whether we need to rebuild predictor matrix

        self.features = {}
        self.comp_features = {}
        self.exclude_first_fix = exclude_first_fix

        if dv_type not in ['fixated', 'count']:
            raise ValueError('Error: unknown DV type specified: "{:s}"!'.format(dv_type))
        self.dv_type = dv_type

        # Make sure imageid is always a chunking variable and all chunk vars exist
        if self._fix._imageid not in chunks:
            chunks.append(self._fix._imageid)
        for var in chunks:
            if var not in self._fix.variables:
                raise ValueError('Error: chunking variable "{:s}" does not exist in dataset!'.format(var))

        self.chunks = chunks
        self.progress = progress

        # Add any specified features to model
        if features:
            if type(features) != list and type(features) != tuple:
                features = [features,]  # force list of features
            for f in features:
                if feature_labels is not None and len(feature_labels) > 0:
                    label = feature_labels.pop(0)
                elif f.label is not None: 
                    label = f.label
                else:
                    label = self._feat_label(f)
                self.add_feature(f, label=label)

            self.update(progress=self.progress)


    def _feat_label(self, feature):
        """ Create label for unlabeled Feature object."""
        cls = feature.__class__.__name__
        label = 'f' + cls[0:5]

        f_label = label
        suffix = 1

        while f_label in self.features.keys():
            f_label = label + str(suffix)
            suffix += 1
        return f_label


    @property
    def predictors(self):
        """ Model predictor matrix, updated if necessary """
        if not self._consistent:
            self.update()
        return self._pred


    def __repr__(self):
        """ String representation for print summary. """
        if not self._consistent:
            self.update()

        r = '<gridfix.FixationModel, {:d} samples, DV={:s}, chunked by: {:s}>\n'.format(self.predictors.shape[0], self.dv_type, str(self.chunks))
        r += 'Fixations:\n\t{:s}\n'.format(str(self._fix))
        r += 'Regions:\n\t{:s}\n'.format(str(self.regionset))

        if len(self.features) > 0:
            r += '\nFeatures:\n'
            for l,f in self.features.items():
                r += '\t{:s}\t{:s}\n'.format(l, f.__class__.__name__)

        if len(self.comp_features) > 0:
            r += '\nFeature Comparisons:\n'
            for l,f in self.comp_features.items():
                r += '{:s}:\n'.format(l)
                for code, feat in f.items():
                    r += '\t{:s}\t{:s}\n'.format(str(code), feat.__class__.__name__)
        return(r)


    def r_source(self, datafile='gridfix.csv', comments=True, scale=True, center=True, 
                 optimizer=None, fixed=None, random=None, random_intercepts=False):
        """ Generate R source code from current feature settings.

        Args:
            datafile (str): predictor matrix file name (for R import via read.table())
            comments (boolean): if True, add explanatory comments and headers to source
            scale (boolean): if True, add code to scale (normalize) feature predictors
            center (boolean): if True, add code to center (demean) feature predictors
            optimizer (str): optional optimizer to pass to R glmer()
            fixed (list): list of column names (strings) to add as fixed factors
            random (list): list of column names (strings) to add as random factors
            random_intercepts (boolean): also add random intercepts

        Returns:
            R source code as string
        """
        r_libs = ['lme4']

        src = ''
        if comments:
            d = time.strftime('%d.%m.%y, %H:%M:%S', time.localtime())
            src =  '# GridFix GLMM R source, generated on {:s}\n'.format(d)
            src += '# input file:\t{:s}\n'.format(datafile)
            src += '# RegionSet:\t{:s}\n'.format(str(self.regionset))
            src += '# DV type:\t{:s}\n'.format(self.dv_type)
            src += '\n'

        # Libraries
        for lib in r_libs:
            src += 'library({:s})\n'.format(lib)
        src += '\n'

        # Predictor file
        src += 'gridfixdata  <- read.table("{:s}", header=T, sep="\\t", row.names=NULL)\n\n'.format(datafile)

        # Factors
        if comments:
            src += '# Define R factors for all chunking variables and group dummy codes\n'
        for chunk in self.chunks:
            src += 'gridfixdata${:s} <- as.factor(gridfixdata${:s})\n'.format(chunk, chunk)
        if len(self.comp_features) > 0:
            for cf in self.comp_features.keys():
                src += 'gridfixdata${:s} <- as.factor(gridfixdata${:s})\n'.format(cf, cf)
        src += '\n'

        # Center and scale
        if scale or center:
            if len(self.features) > 0:
                r_cent = 'FALSE'
                r_scal = 'FALSE'
                if center:
                    r_cent = 'TRUE'
                if scale:
                    r_scal = 'TRUE'
                if comments:
                    src += '# Center and scale predictors\n'
                for f in self.features.keys():
                    src += 'gridfixdata${:s}_C <- scale(gridfixdata${:s}, center={:s}, scale={:s})\n'.format(f, f, r_cent, r_scal)
                src += '\n'

        # GLMM model formula
        formula = 'dvFix ~ 1'

        if fixed is None:
            # Best guess: all simple features should be fixed factors!
            if len(self.features) > 0:
                fixed = self.features.keys()
            else:
                fixed = []

        fixed_vars = ''
        for f in fixed:
            if scale or center:
                fixed_vars += ' + {:s}_C '.format(f)
            else:
                fixed_vars += ' + {:s} '.format(f)
        formula += fixed_vars

        if random is None:
            # imageid should be a random factor by default
            random = [self._fix._imageid]
        
        if len(fixed) > 0:
            for r in random:
                if random_intercepts:
                    formula += ' + (1{:s} | {:s})'.format(fixed_vars, r)
                else:
                    formula += ' + (1 | {:s})'.format(r)

        # Optimizer parameter
        opt_call = ''
        if optimizer is not None:
            opt_call = ', control=glmerControl(optimizer="{:s}")'.format(optimizer)

        # Model family based on DV type
        if self.dv_type == 'fixated':
            model_fam = 'binomial'
        elif self.dv_type == 'count':
            model_fam = 'poisson'

        # GLMM model call
        if comments:
            src += '# NOTE: this source code can only serve as a scaffolding for your own analysis!\n'
            src += '# You MUST adapt the GLMM model formula below to your model, then uncomment the corresponding line!\n'
            src += '#'
        src += 'model <- glmer({:s}, data=gridfixdata{:s}, family={:s})\n\n'.format(formula, opt_call, model_fam)

        out_f, ext = os.path.splitext(datafile)
        src += 'save(file="{}_GLMM.Rdata", list = c("model"))\n\n'.format(out_f)
        src += 'print(summary(model))\n'
        return src


    def _process_chunk(self, chunk_vals, data, pred_columns, group_levels):
        """ Process a single data chunk. """
        if data is not None:
            sel_labels = dict(zip(self.chunks, chunk_vals))
            imageid = str(sel_labels[self._fix._imageid])
            tmpdf = DataFrame(columns=pred_columns, index=range(len(self.regionset[imageid])))

            # groupby returns a single string if only one chunk columns is selected
            if type(chunk_vals) != tuple:
                chunk_vals = (chunk_vals,)

            # Fixations
            subset = self._fix.select_fix(sel_labels)

            # Chunk and region values
            for col in self.chunks:
                tmpdf[col] = data[col].iloc[0]

            # Region ID and numbering
            tmpdf.regionid = np.array(self.regionset.info[self.regionset.info.imageid == imageid].regionid, dtype=str)
            tmpdf.regionno = np.array(self.regionset.info[self.regionset.info.imageid == imageid].regionno, dtype=int)

            # Fixated and non-fixated regions
            if self.dv_type == 'fixated':
                tmpdf['dvFix'] = self.regionset.fixated(subset, imageid=imageid, exclude_first=self.exclude_first_fix)
            elif self.dv_type == 'count':
                tmpdf['dvFix'] = self.regionset.fixated(subset, imageid=imageid, exclude_first=self.exclude_first_fix, count=True)

            # Simple per-image features
            for feat_col, feat in self.features.items():
                tmpdf[feat_col] = feat.apply(imageid)

            # Feature group comparisons
            if len(self.comp_features) > 0:
                for levels in group_levels:
                    for idx, gc in enumerate(self.comp_features.keys()):
                        tmpdf[gc] = levels[idx]
                        tmpdf['{:s}_val'.format(gc)] = self.comp_features[gc][levels[idx]].apply(imageid)

            return tmpdf


    def update(self, progress=False):
        """ Update predictor matrix from features (this may take a while).

        Args:
            progress (boolean): if True, print model creation progress
        """
        ts = time.time()

        # Output DF columns
        pred_columns = self.chunks + ['regionid', 'regionno', 'dvFix']
        pred_columns += list(self.features.keys())
        for cf in self.comp_features.keys():
            pred_columns += [cf, '{:s}_val'.format(cf)]

        pred_new = DataFrame(columns=pred_columns)

        splitdata = self._fix.data.groupby(self.chunks)

        group_levels = []
        if len(self.comp_features) > 0:
            groups = [list(f.keys()) for i,f in self.comp_features.items()]
            group_levels = list(itertools.product(*groups))

        # Process individual chunks
        for chunk_vals, data in splitdata:
            if progress:
                print(chunk_vals)
            results = self._process_chunk(chunk_vals, data, pred_columns, group_levels)
            pred_new = pred_new.append(results)

        self._pred = pred_new
        self._consistent = True
        self.runtime = time.time() - ts     # rebuild duration
        

    def add_feature(self, feature, label=None):
        """ Add a feature to the model.

        Args:
            feature (Feature): Feature object to add
            label (str): label and output column name for this feature
        """
        # Generate unique feature label
        if label is None:
            label = self._feat_label(feature)

        # Check feature length
        if len(feature) != len(self.regionset):
            w = 'Could not add feature "{:s}": invalid length ({:d} instead of {:d})!'
            raise ValueError(label, len(feature), len(self.regionset))

        self.features[label] = feature
        self._consistent = False


    def add_comparison(self, features, codes=None, label=None):
        """ Add a feature comparison group to the model.

        This generates a long-style predictor matrix for the specified features,
        needed to compare e.g. different saliency maps in their relative effects.

        Args:
            features (list): list of Feature objects to combine into a group
            codes (list): numeric codes to use in "dummy coding", e.g. [0, 1, 2]
            label (str): label and output column name for this feature group
        """
        # Generate unique group label
        if label is None:
            suffix = 1
            g_label = 'fC' + str(suffix)

            while g_label in self.comp_features.keys():
                suffix += 1
                g_label = 'fC' + str(suffix)

        # Generate dummy codes if necessary
        if codes is None:
            codes = range(0, len(features))
        else:
            codes = [int(c) for c in codes]

        comp = {codes[c]: f for c,f in enumerate(features)}
        self.comp_features[g_label] = comp
        self._consistent = False


    def save(self, basename, sep='\t', pred=True, pred_pickle=False, src=True, src_comments=True, precision=10,
             optimizer=None, fixed=None, random=None, random_intercepts=False):
        """ Saves the predictor matrix to a CSV text file.

        Args:
            basename (str): base filename to save to, without extension
            sep (str): item separator, default TAB
            pred (boolean): if True, output predictor matrix as CSV
            pred_pickle (boolean): if True, also save predictors to Pickle object
            src (boolean): if True, output r source code file for lme4
            src_comments (boolean): if True, add comments to source code
            precision (int): number of decimal places for CSV (default: 10)
            optimizer (str): optional optimizer to pass to R glmer()
            fixed (list): list of column names (strings) to add as fixed factors
            random (list): list of column names (strings) to add as random factors
            random_intercepts (boolean): also add random intercepts

        """
        if not self._consistent:
            self.update()

        if pred:
            f_pred = '{:s}.csv'.format(basename)
            self.predictors.to_csv(f_pred, sep, index=False, float_format='%.{:d}f'.format(precision))

        if pred_pickle:
            f_pred = '{:s}.pkl'.format(basename)
            self.predictors.to_pickle(f_pred)

        if src:
            f_src = '{:s}.R'.format(basename)
            src = self.r_source(comments=src_comments, datafile=f_pred, optimizer=optimizer,
                                fixed=fixed, random=random, random_intercepts=random_intercepts)
            with open(f_src, 'w') as sf:
                sf.write(src)


if __name__ == '__main__':
    print('The gridfix modules cannot be called directly. Please use one of the included tools, e.g. gridmap.')