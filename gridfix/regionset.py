#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np
import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from PIL import Image

from fractions import Fraction
from pandas import DataFrame, read_table

from .model import ImageSet


class RegionSet(object):
    """ Base class for sets of image regions of interest.

    RegionSets can be used to spatially group fixations, create Feature objects
    for a FixationModel and split an image into parts. Classes inheriting from 
    RegionSet may specify functions to create regions.

    Attributes:
        info (DataFrame): table of region metadata (labels, bboxes, number of pixels...)
        imageids (list): list of all imageids associated with this RegionSet
        is_global (bool): True if regions are global (non-image-specific)
        label (str): optional label to distinguish between RegionSets.
        memory_usage (float): memory usage of all binary masks (kiB)
        size (tuple): image dimensions, specified as (width, height).
    """

    def __init__(self, size, regions, region_labels=None, label=None):
        """ Create a new RegionSet from existing region masks.

        Args:
            size (tuple): image dimensions, specified as (width, height)
            regions: 3d ndarray (bool) with global set of masks 
                OR dict of multiple such ndarrays, with imageids as keys
            region_labels: list of region labels IF _regions_ is a single array
                OR dict of such lists, with imageids as keys
            label (str): optional descriptive label for this RegionSet

        Raises:
            ValueError if incorrectly formatted regions/region_labels provided
        """
        self._regions = {'*': np.ndarray((0,0,0))}
        self._labels  = {'*': []}

        self.size = size
        self.label = label
        self._msize = (size[1], size[0])   # matrix convention

        if isinstance(regions, dict):
            # Dict with image-specific region ndarrays
            self._regions = regions
            if region_labels is not None and isinstance(region_labels, dict) and len(regions) == len(region_labels):
                # Check imageids for consistency
                for r in regions.keys():
                    if r not in region_labels.keys():
                        raise ValueError('Labels not consistent: {:s} not in region_labels'.format(r))
                for r in region_labels.keys():
                    if r not in regions.keys():
                        raise ValueError('Labels not consistent: {:s} not in regions'.format(r))
                self._labels = region_labels
            else:
                self._labels = {}
                for imid in regions:
                    self._labels[imid] = [str(x+1) for x in range(len(regions[imid]))]

        elif isinstance(regions, np.ndarray):
            # Single array of regions - assume global region set ('*')
            if regions.shape[1:] == self._msize:
                self._regions['*'] = regions.astype(bool)
            if region_labels is not None and len(region_labels) == regions.shape[0]:
                self._labels['*'] = region_labels
            else:
                self._labels['*'] = [str(x+1) for x in range(regions.shape[0])]

        else:
            raise ValueError('First argument for RegionSet creation must be ndarray ' + 
                             '(global regions) or dict of ndarrays (image-specific regions)!')

        self.info = self._region_metadata()


    def __repr__(self):
        """ String representation """
        r = 'gridfix.RegionSet(label={:s}, size=({:d}, {:d}),\nregions={:s},\nregion_labels={:s})'
        return r.format(str(self.label), self.size[0], self.size[1], str(self._regions), str(self._labels))


    def __str__(self):
        """ Short string representation for printing """
        r = '<{:s}{:s}, size={:s}, {:d} region{:s}{:s}, memory={:.1f} kB>'
        
        myclass = str(self.__class__.__name__)
        if self.label is not None:
            lab = ' ({:s})'.format(self.label)
        else:
            lab = ''

        num_s = ''
        num_r = len(self)
        if num_r > 1:
            num_s = 's'

        imid_s = ''
        if len(self._regions) > 1 and not self.is_global:
            imid_s = ' in {:d} images'.format(len(self._regions))

        return r.format(myclass, lab, str(self.size), num_r, num_s, imid_s, self.memory_usage)


    def __len__(self):
        """ Overload len(RegionSet) to report total number of regions. """
        if self.is_global:                
            return len(self._regions['*'])
        else:
            num_r = 0
            for imid in self._regions:
                num_r += len(self._regions[imid])
            return num_r


    def __getitem__(self, imageid):
        """ Bracket indexing returns all region masks for a specified imageid.
            If global regions are set ('*'), always return global region set. 
        """
        return self._select_region(imageid)


    def _region_metadata(self):
        """ Return DataFrame of region metadata """
        info_cols = ['imageid', 'region', 'left', 'top', 'right', 'bottom', 'width', 'height', 'area', 'imgfrac']
        info = []

        if self.is_global:
            imageids = ['*']
        else:
            imageids = self.imageids

        for imid in imageids:
            reg = self._select_region(imid)
            lab = self._select_labels(imid)

            for i,l in enumerate(lab):
                a = np.argwhere(reg[i])
                (top, left) = a.min(0)[0:2]
                (bottom, right) = a.max(0)[0:2]

                area = reg[i][reg[i] > 0].sum()
                imgfrac = round(area / (reg[i].shape[0] * reg[i].shape[1]), 4)
                rmeta = [imid, l, left, top, right, bottom, right-left+1, bottom-top+1, area, imgfrac]
                info.append(rmeta)

        return DataFrame(info, columns=info_cols)


    def _select_region(self, imageid=None):
        """ Select region by imageid with consistency check """
        if self.is_global:
            return(self._regions['*'])

        if imageid is not None and imageid in self._regions.keys():
            return(self._regions[imageid])
        else:
            raise ValueError('RegionSet contains image-specific regions, but no valid imageid was specified!')


    def _select_labels(self, imageid=None):
        """ Select region labels corresponding to _select_region """
        if self.is_global:
            return(self._labels['*'])

        if imageid is not None and imageid in self._regions.keys():
            return(self._labels[imageid])
        else:
            raise ValueError('RegionSet contains image-specific regions, but no valid imageid was specified!')


    @property
    def is_global(self):
        """ Return True if a global map is defined (key '*') """
        if '*' in self._regions.keys():
            return True
        else:
            return False


    @property
    def imageids(self):
        """ Return list of imageids for which region maps exist """
        if self.is_global:
            return []
        
        imids = []
        for imid in self._regions.keys():
            imids.append(imid)
        return imids


    @property
    def memory_usage(self):
        """ Calculate size in memory of all regions combined """
        msize = 0.0
        for reg in self._regions.keys():
            msize += float(self._regions[reg].nbytes) / 1024.0
        return msize


    def count_map(self, imageid=None):
        """ Return the number of regions referencing each pixel.

        Args:
            imageid (str):  if set, return map for specified image only 

        Returns:
            2d ndarray of image size, counting number of regions for each pixel
        """
        
        cm = np.zeros(self._msize, dtype=int)

        if self.is_global:
            for re in self._regions['*'][:, ...]:
                cm += re.astype(int)
            return cm

        elif imageid is None:
            for imid in self._regions:
                if imid == '*':
                    continue
                for re in self._regions[imid][:, ...]:
                    cm += re.astype(int)

        else:
            r = self._select_region(imageid)
            for re in r[:, ...]:
                cm += re.astype(int)

        return cm

    
    def mask(self, imageid=None):
        """ Return union mask of all regions or regions for specified image.

        Args:
            imageid (str):  if set, return mask for specified image only

        Returns:
            2d ndarray of image size (bool), True where at least one region
            references the corresponding pixel.
        """
        return self.count_map(imageid).astype(bool)


    def region_map(self, imageid=None):
        """ Return map of region numbers, global or image-specifid.

        Args:
            imageid (str):  if set, return map for specified image only

        Returns:
            2d ndarray (int), containing the number (ascending) of the last
            region referencing the corresponding pixel.
        """
        apply_regions = self._select_region(imageid)
        tmpmap = np.zeros(self._msize)
        for idx, region in enumerate(apply_regions):
            tmpmap[region] = (idx + 1)
        return tmpmap


    def coverage(self, imageid=None, normalize=False):
        """ Calculates coverage of the total image size as a scalar.

        Args:
            imageid (str):  if set, return coverage for specified image only 
            normalize (bool): if True, divide global result by number of imageids in set.

        Returns:
            Total coverage as a floating point number.
        """
        if imageid is not None:
            counts = self.count_map(imageid)
            cov = float(counts.sum()) / float(self.size[0] * self.size[1])
            return cov
        else:
            # Global coverage for all imageids
            cm = np.zeros(self._msize, dtype=int)
            for re in self._regions.keys():
                if re == '*':
                    cm += self.count_map('*')
                    break
                cm += self.count_map(re)

            cov = float(cm.sum()) / float(self.size[0] * self.size[1])
            if normalize:
                cov = cov / len(self)
            return cov


    def plot(self, imageid=None, values=None, cmap=None, image_only=False, ax=None, alpha=1.0):
        """ Plot regions as map of shaded areas with/without corresponding feature values

        Args:
            imageid (str): if set, plot regions for specified image
            values (array-like): one feature value per region
            cmap (str): name of matplotlib colormap to use
            image_only (boolean): if True, return only image content without axes
            ax (Axes): axes object to draw to, to include result in other figure
            alpha (float): opacity of plotted regions (set < 1 to visualize overlap)
        
        Returns:
            matplotlib figure object, or None if passed an axis to draw on
        """
        apply_regions = self._select_region(imageid)
        tmpmap = np.zeros(self._msize)

        if ax is not None:
            ax1 = ax
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)

        if cmap is None:
            if values is None and 'viridis' in plt.colormaps():
                cmap = 'viridis'
            else:
                cmap = 'gray'

        if alpha < 1.0:
            # allow stacking by setting masked values transparent
            alpha_cmap = plt.get_cmap(cmap)
            alpha_cmap.set_bad(alpha=0)

            ax1.imshow(tmpmap, cmap=plt.get_cmap('gray'), interpolation='none')

            for idx, region in enumerate(apply_regions):
                rmap = np.zeros(self._msize)
                if values is not None and len(values) == apply_regions.shape[0]:
                    rmap[region] = values[idx]
                    ax1.imshow(np.ma.masked_equal(rmap, 0), cmap=alpha_cmap, interpolation='none', alpha=alpha, 
                               vmin=min(values), vmax=max(values))

                else:
                    rmap[region] = idx + 1
                    ax1.imshow(np.ma.masked_equal(rmap, 0), cmap=alpha_cmap, interpolation='none', alpha=alpha, 
                               vmin=0, vmax=apply_regions.shape[0])

        else:
            # If no alpha requested, this is much faster but doesn't show overlap
            ax1.imshow(tmpmap, cmap=plt.get_cmap('gray'), interpolation='none')
            
            if values is not None and len(values) == apply_regions.shape[0]:
                rmap = np.zeros(self._msize)
                for idx, region in enumerate(apply_regions):
                    rmap[region] = values[idx]
                ax1.imshow(np.ma.masked_equal(rmap, 0), cmap=plt.get_cmap(cmap), interpolation='none', vmin=min(values), vmax=max(values))
            else:
                ax1.imshow(np.ma.masked_equal(self.region_map(imageid), 0), cmap=plt.get_cmap(cmap), interpolation='none', 
                           vmin=0, vmax=apply_regions.shape[0])

        if image_only:
            ax1.axis('off')
        
        else:
            t = '{:s}'.format(self.__class__.__name__)
            if self.label is not None:
                t += ' "{:s}"'.format(self.label)
            if imageid is not None:
                t += ': {:s}'.format(imageid)
            ax1.set_title(t)

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig


    def plot_regions_on_image(self, imageid=None, imageset=None, cmap=None, fill=False,
                              alpha=0.4, labels=False, image_only=False, ax=None):
        """ Plot region bounding boxes on corresponding image

        Args:
            imageid (str): if set, plot regions for specified image
            imageset (ImageSet): ImageSet object containing background image/map
            cmap (str): name of matplotlib colormap to use for boundin boxes
            fill (boolean): draw shaded filled rectangles instead of boxes
            alpha (float): rectangle opacity (only when fill=True)
            labels (boolean): if True, draw text labels next to regions
            image_only (boolean): if True, return only image content without axes
            ax (Axes): axes object to draw to, to include result in other figure

        Returns:
            matplotlib figure object, or None if passed an axis to draw on
        """
        if imageset is None or imageid not in imageset.imageids:
            raise ValueError('To plot regions on top of image, specify ImageSet containing corresponding background image!')

        if ax is not None:
            ax1 = ax
        else:
            fig = plt.figure()
            ax1 = fig.add_subplot(1,1,1)

        ax1.imshow(imageset[imageid], cmap=plt.get_cmap('gray'), interpolation='none')

        if cmap is None:
            if 'viridis' in plt.colormaps():
                cmap = 'viridis'
            else:
                cmap = 'hsv'

        boxcolors = plt.get_cmap(cmap)
        cstep = 0

        if self.is_global:
            rmeta = self.info[self.info.imageid == '*']
        else:
            rmeta = self.info[self.info.imageid == imageid]

        for idx, region in rmeta.iterrows():
            c = boxcolors(cstep/len(rmeta))
            cstep += 1
            if not fill:
                ax1.add_patch(Rectangle((region.left, region.top), region.width, region.height, color=c, fill=False, linewidth=2))
            else:
                ax1.add_patch(Rectangle((region.left, region.top), region.width, region.height, color=c, linewidth=0, alpha=0.7))
            if labels:
                # Draw text labels with sensible default positions
                if region.right > (self.size[0] * .95):
                    tx = region.right
                    ha = 'right'
                else:
                    tx = region.left
                    ha = 'left'
                if region.bottom > (self.size[1] * .95):
                    ty = region.top - 5
                else:
                    ty = region.bottom + 20
                ax1.text(tx, ty, region.region, horizontalalignment=ha)

        if image_only:
            ax1.axis('off')

        else:
            t = '{:s}'.format(self.__class__.__name__)
            if self.label is not None:
                t += ' "{:s}"'.format(self.label)
            t += ': {:s}'.format(imageid)
            ax1.set_title(t)

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig


    def apply(self, image, imageid=None, crop=False):
        """ Apply this RegionSet to a specified image.

        Returns a list of the image arrays "cut out" by each region mask, with
        non-selected image areas in black. If regionset is not global, _imageid_ needs
        to be specified!

        Args:
            image (ndarray): image array to be segmented.
            imageid (str): valid imageid (to select image-specific regions if not a global regionset)
            crop (bool): if True, return image cropped to bounding box of selected area
        
        Returns:
            If crop=False, a list of ndarrays of same size as image, with non-selected areas
            zeroed. Else a list of image patch arrays cropped to bounding box size.
        """
        slices = []
        apply_regions = self._select_region(imageid)

        for region in apply_regions:
            mask = (region == True)
            out = np.zeros(image.shape)
            out[mask] = image[mask]

            if crop:
                a = np.argwhere(out)
                (ul_x, ul_y) = a.min(0)[0:2]
                (br_x, br_y) = a.max(0)[0:2]
                out = out[ul_x:br_x+1, ul_y:br_y+1]
            slices.append(out)

        return slices


    def export_patches(self, image, imageid=None, crop=True, image_format='png', rescale=False):
        """ Apply this RegionSet to an image array and save the resulting image patches as files.

        Saves an image of each image part "cut out" by each region mask, cropped by default.
        If the RegionSet is not global, imageid needs to be specified!

        Args:
            image (ndarray): image array to be segmented.
            imageid (str): imageid (to select image-specific regions if not a global regionset)
            crop (bool): if True, return image cropped to bounding box of selected area
            image_format (str): image format that PIL understands (will also be used for extension)
            rescale (bool): if True, scale pixel values to full 0..255 range
                before saving (e.g., for saliency maps)
        """
        apply_regions = self._select_region(imageid)
        apply_labels = self._select_labels(imageid)
        imstr = '{:s}_{:s}.{:s}'

        for idx, region in enumerate(apply_regions):
            mask = (region == True)
            out = np.zeros(image.shape)
            out[mask] = image[mask]

            if crop:
                a = np.argwhere(out)
                (ul_x, ul_y) = a.min(0)[0:2]
                (br_x, br_y) = a.max(0)[0:2]
                out = out[ul_x:br_x+1, ul_y:br_y+1]

            if imageid is None or imageid == '*':
                imageid = 'image'

            if rescale:
                out = (out - out.min()) / out.max() * 255.0
            else:
                out *= 255.0

            rimg = Image.fromarray(np.array(out, np.uint8))
            rimg.save(imstr.format(imageid, apply_labels[idx], image_format), image_format)


    def export_patches_from_set(self, imageset, crop=True, image_format='png', rescale=False):
        """ Save all sliced image patches from an ImageSet as image files. 

        Saves an image of each image part "cut out" by each region mask, cropped by default.
        If the RegionSet is not global, only images with valid region masks will be processed.

        Args:
            imageset (ImageSet): a valid ImageSet containing images to slice
            imageid (str): imageid (to select image-specific regions if not a global regionset)
            crop (bool): if True, return image cropped to bounding box of selected area
            image_format (str): image format that PIL understands (will also be used for extension)
            rescale (bool): if True, scale pixel values to full 0..255 range
                before saving (e.g., for saliency maps)
        """
        if not isinstance(imageset, ImageSet):
            raise TypeError('First argument must be an ImageSet! To slice a single image, use export_patches().')

        for cimg in imageset.imageids:
            if not self.is_global and cimg not in self.imageids:
                print('Warning: RegionSet contains image-specific regions, but no regions available for {:s}. Skipped.'.format(cimg))
            else:
                self.export_patches(imageset[cimg], imageid=cimg, crop=crop, image_format=image_format, rescale=rescale)
            

    def fixated(self, fixations, imageid=None, count=False, exclude_first=False):
        """ Returns visited / fixated regions using data from a Fixations object.

        Args:
            fixations (Fixations/DataFrame): fixation data to test against regions
            imageid (str): imageid (to select image-specific regions if not a global regionset)
            count (bool): if True, return number of fixations per region instead of boolean values
            exclude_first (bool): if True, first fixated region will always be returned as NaN

        Returns:
            1D ndarray (float) containing number of fixations per region (if count=True) 
            or the values 0.0 (region was not fixated) or 1.0 (region was fixated)
        """
        apply_regions = self._select_region(imageid)
        vis = np.zeros(apply_regions.shape[0], dtype=float)
        
        # Drop out-of-bounds fixations
        fix = fixations.data[(fixations.data[fixations._xpx] >= 0) & 
                             (fixations.data[fixations._xpx] < self.size[0]) & 
                             (fixations.data[fixations._ypx] >= 0) & 
                             (fixations.data[fixations._ypx] < self.size[1])]
    
        if len(fix) > 0:
            first_fix = fixations.data[fixations.data[fixations._fixid] == min(fixations.data[fixations._fixid])]
            if len(first_fix) > 1 and exclude_first:
                print('Warning: you have requested to drop the first fixated region, but more than one ' +
                      'location ({:d}) matches the lowest fixation ID! Either your fixation ' .format(len(first_fix)) +
                      'IDs are not unique or the passed dataset contains data from multiple images or conditions.')

            for (idx, roi) in enumerate(apply_regions):
                fv = roi[fix[fixations._ypx], fix[fixations._xpx]]
                if isinstance(fv, np.ndarray):
                    num_fix = sum(fv)
                    vis[idx] = num_fix

                if exclude_first:
                    is_first = roi[first_fix[fixations._ypx], first_fix[fixations._xpx]]
                    if isinstance(is_first, np.ndarray) and any(is_first):
                        vis[idx] = np.nan
                    elif is_first:
                        vis[idx] = np.nan

        if not count:
            vis[vis >= 1.0] = 1.0
            vis[vis < 1.0] = 0.0

        return vis



class GridRegionSet(RegionSet):
    """ RegionSet defining an n-by-m regular grid covering the full image size.

    Attributes:
        cells (list): list of bounding box tuples for each cell,
            each formatted as (left, top, right, bottom)
        gridsize (tuple): grid dimensions as (width, height). If unspecified,
            gridfix will try to choose a sensible default.
        label (string): optional label to distinguish between RegionSets
    """

    def __init__(self, size, gridsize=None, label=None, region_labels=None):
        """ Create a new grid RegionSet

        Args:
            size (tuple): image dimensions, specified as (width, height).
            gridsize(tuple): grid dimensions, specified as (width, height).
            region_labels (string): list of optional region labels (default: cell#)
        """

        if gridsize is None:
            gridsize = self._suggest_grid(size)
            print('Note: no grid size was specified. Using {:d}x{:d} based on image size.'.format(gridsize[0], gridsize[1]))

        (regions, cells) = self._grid(size, gridsize)
        RegionSet.__init__(self, size=size, regions=regions, label=label, region_labels=region_labels)

        self.gridsize = gridsize

        # List of region bboxes
        self.cells = cells


    def __str__(self):
        """ Short string representation for printing """
        r = '<gridfix.GridRegionSet{:s}, size={:s}, {:d}x{:d} grid, {:d} cell{:s}, memory={:.1f} kB>'
        
        if self.label is not None:
            lab = ' ({:s})'.format(self.label)
        else:
            lab = ''

        num_s = ''
        num_r = len(self)
        if num_r > 1:
            num_s = 's'
        return r.format(lab, str(self.size), self.gridsize[0], self.gridsize[1], num_r, 
                        num_s, self.memory_usage)


    def _suggest_grid(self, size):
        """ Suggest grid dimensions based on image size.

        Args:
            size (tuple): image dimensions, specified as (width, height).

        Returns:
            Suggested grid size tuple as (width, height).
        """
        aspect = Fraction(size[0], size[1])
        s_width = aspect.numerator
        s_height = aspect.denominator
        if s_width < 6:
            s_width *= 2
            s_height *= 2
        return (s_width, s_height)


    def _grid(self, size, gridsize):
        """ Build m-by-n (width,height) grid as 3D nparray.

        Args:
            size (tuple): image dimensions, specified as (width, height).
            gridsize(tuple): grid dimensions, specified as (width, height).

        Returns:
            tuple containing the grid regions and their bounding box coordinates
            as (grid, cells):

                grid (numpy.ndarray): regions for RegionSet creation
                cells (list): list of bounding box tuples for each cell, 
                    each formatted as (left, top, right, bottom)

        """
        (width, height) = size
        _msize = (size[1], size[0])
        cell_x = int(width / gridsize[0])
        cell_y = int(height / gridsize[1])
        n_cells = int(gridsize[0] * gridsize[1])

        grid = np.zeros((n_cells,) + _msize, dtype=bool)
        cells = []

        # Sanity check: do nothing if image dimensions not cleanly divisible by grid
        if width % gridsize[0] > 0 or height % gridsize[1] > 0:
            e = 'Error: image dimensions not cleanly divisible by grid! image=({:d}x{:d}), grid=({:d}x{:d})'
            raise ValueError(e.format(width, height, gridsize[0], gridsize[1]))

        # Create a mask of 1s/True for each cell
        cellno = 0
        for y_es in range(0, height, cell_y):
            for x_es in range(0, width, cell_x):
                mask = np.zeros(_msize, dtype=bool)
                mask[y_es:y_es + cell_y, x_es:x_es + cell_x] = True
                grid[cellno,...] = mask
                cells.append((x_es, y_es, x_es + cell_x, y_es + cell_y))
                cellno += 1

        return (grid, cells)
