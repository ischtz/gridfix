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

    def __init__(self, size, regions, region_labels=None, label=None, add_background=False):
        """ Create a new RegionSet from existing region masks.

        Args:
            size (tuple): image dimensions, specified as (width, height)
            regions: 3d ndarray (bool) with global set of masks 
                OR dict of multiple such ndarrays, with imageids as keys
            region_labels: list of region labels IF _regions_ is a single array
                OR dict of such lists, with imageids as keys
            label (str): optional descriptive label for this RegionSet
            add_background (bool): if True, this creates a special region to capture all
                fixations that don't fall on an explicit region ("background" fixations)

        Raises:
            ValueError if incorrectly formatted regions/region_labels provided
        """
        self._regions = {'*': np.ndarray((0,0,0))}
        self._labels  = {'*': []}

        self.size = size
        self.label = label
        self._msize = (size[1], size[0])   # matrix convention
        self.has_background = False

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

        if add_background:
            for iid in self._regions.keys():
                bgmask = ~self.mask(iid).reshape(1, size[1], size[0])
                self._regions[iid] = np.concatenate([self._regions[iid], bgmask], axis=0)
                self._labels[iid].append('__BG__')
            self.has_background = True

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
        info_cols = ['imageid', 'regionid', 'regionno', 'left', 'top', 'right', 'bottom', 'width', 'height', 'area', 'imgfrac']
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
                if a.shape[0] > 0:
                    (top, left) = a.min(0)[0:2]
                    (bottom, right) = a.max(0)[0:2]
                    (width, height) = (right-left+1, bottom-top+1)
                    area = reg[i][reg[i] > 0].sum()
                    imgfrac = round(area / (reg[i].shape[0] * reg[i].shape[1]), 4)
                else:
                    # Region is empty - shouldn't, but can happen with add_background at full coverage
                    (top, left, bottom, right, width, height, area, imgfrac) = (0,) * 8

                rmeta = [imid, l, i+1, left, top, right, bottom, width, height, area, imgfrac]
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


    def count_map(self, imageid=None, ignore_background=True):
        """ Return the number of regions referencing each pixel.

        Args:
            imageid (str):  if set, return map for specified image only 
            ignore_background (bool): if True, ignore auto-generated background region

        Returns:
            2d ndarray of image size, counting number of regions for each pixel
        """
        
        cm = np.zeros(self._msize, dtype=int)

        if self.is_global:
            for reidx, re in enumerate(self._regions['*'][:, ...]):
                if ignore_background and self._labels['*'][reidx] == '__BG__':
                    continue
                cm += re.astype(int)
            return cm

        elif imageid is None:
            for imid in self._regions:
                if imid == '*':
                    continue
                for reidx, re in enumerate(self._regions[imid][:, ...]):
                    if ignore_background and self._labels[imid][reidx] == '__BG__':
                        continue
                    cm += re.astype(int)

        else:
            r = self._select_region(imageid)
            l = self._select_labels(imageid)
            for reidx, re in enumerate(r[:, ...]):
                if ignore_background and l[reidx] == '__BG__':
                    continue
                cm += re.astype(int)

        return cm

    
    def mask(self, imageid=None, ignore_background=True):
        """ Return union mask of all regions or regions for specified image.

        Args:
            imageid (str):  if set, return mask for specified image only
            ignore_background (bool): if True, ignore auto-generated background region

        Returns:
            2d ndarray of image size (bool), True where at least one region
            references the corresponding pixel.
        """
        return self.count_map(imageid, ignore_background).astype(bool)


    def region_map(self, imageid=None, ignore_background=True):
        """ Return map of region numbers, global or image-specifid.

        Args:
            imageid (str):  if set, return map for specified image only
            ignore_background (bool): if True, ignore auto-generated background region

        Returns:
            2d ndarray (int), containing the number (ascending) of the last
            region referencing the corresponding pixel.
        """
        apply_regions = self._select_region(imageid)
        apply_labels = self._select_labels(imageid)
        tmpmap = np.zeros(self._msize)
        for idx, region in enumerate(apply_regions):
            if ignore_background and apply_labels[idx] == '__BG__':
                continue
            tmpmap[region] = (idx + 1)
        return tmpmap


    def coverage(self, imageid=None, normalize=False, ignore_background=True):
        """ Calculates coverage of the total image size as a scalar.

        Args:
            imageid (str):  if set, return coverage for specified image only 
            normalize (bool): if True, divide global result by number of imageids in set.
            ignore_background (bool): if True, ignore auto-generated background region

        Returns:
            Total coverage as a floating point number.
        """
        if imageid is not None:
            counts = self.count_map(imageid, ignore_background)
            cov = float(counts.sum()) / float(self.size[0] * self.size[1])
            return cov
        else:
            # Global coverage for all imageids
            cm = np.zeros(self._msize, dtype=int)
            for re in self._regions.keys():
                if re == '*':
                    cm += self.count_map('*', ignore_background)
                    break
                cm += self.count_map(re, ignore_background)

            cov = float(cm.sum()) / float(self.size[0] * self.size[1])
            if normalize:
                cov = cov / len(self)
            return cov


    def plot(self, imageid=None, values=None, cmap=None, image_only=False, ax=None, alpha=1.0):
        """ Plot regions as map of shaded areas with/without corresponding feature values

        Args:
            imageid (str): if set, plot regions for specified image
            values (array-like): one feature value per region
            cmap (str): name of matplotlib colormap to use to distinguish regions
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

        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)

        if alpha < 1.0:
            # allow stacking by setting masked values transparent
            alpha_cmap = cmap
            alpha_cmap.set_bad(alpha=0)

            ax1.imshow(tmpmap, cmap=plt.get_cmap('gray'), interpolation='none')

            for idx, region in enumerate(apply_regions):
                rmap = np.zeros(self._msize)
                if values is not None and len(values) == apply_regions.shape[0]:
                    rmap[region] = values[idx]
                    ax1.imshow(np.ma.masked_equal(rmap, 0), cmap=alpha_cmap, interpolation='none', alpha=alpha, 
                               vmin=0, vmax=np.nanmax(values))

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
                ax1.imshow(np.ma.masked_equal(rmap, 0), cmap=cmap, interpolation='none', vmin=0, vmax=np.nanmax(values))
            else:
                ax1.imshow(np.ma.masked_equal(self.region_map(imageid), 0), cmap=cmap, interpolation='none',
                           vmin=0, vmax=apply_regions.shape[0])

        if image_only:
            ax1.axis('off')
        
        else:
            t = '{:s}'.format(self.__class__.__name__)
            if self.label is not None:
                t += ': {:s}'.format(self.label)
            if imageid is not None:
                t += ' (img: {:s})'.format(imageid)
            ax1.set_title(t)

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig


    def plot_regions_on_image(self, imageid=None, imageset=None, image_cmap=None, cmap=None, plotcolor=None, 
                              fill=False, alpha=0.4, labels=False, image_only=False, ax=None):
        """ Plot region bounding boxes on corresponding image

        Args:
            imageid (str): if set, plot regions for specified image
            imageset (ImageSet): ImageSet object containing background image/map
            image_cmap (str): name of matplotlib colormap to use for image
            cmap (str): name of matplotlib colormap to use for bounding boxes
            plotcolor (color): matplotlib color for bboxes (overrides colormap)
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

        if image_cmap is not None:
            if type(image_cmap) == str:
                image_cmap = plt.get_cmap(image_cmap)
            ax1.imshow(imageset[imageid], cmap=image_cmap, interpolation='none')
        else:
            ax1.imshow(imageset[imageid], interpolation='none')

        if cmap is None:
            if 'viridis' in plt.colormaps():
                cmap = 'viridis'
            else:
                cmap = 'hsv'

        if type(cmap) == str:
            boxcolors = plt.get_cmap(cmap)
        else:
            boxcolors = cmap
        cstep = 0

        if self.is_global:
            rmeta = self.info[self.info.imageid == '*']
        else:
            rmeta = self.info[self.info.imageid == imageid]

        for idx, region in rmeta.iterrows():
            if self.has_background and region.regionid == '__BG__':
                # Always skip background region when drawing bboxes
                continue
            if plotcolor is None:
                c = boxcolors(cstep/len(rmeta))
            else:
                c = plotcolor
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
                ax1.text(tx, ty, region.regionid, horizontalalignment=ha)

        if image_only:
            ax1.axis('off')

        else:
            t = '{:s}'.format(self.__class__.__name__)
            if self.label is not None:
                t += ': {:s}'.format(self.label)
            t += ' (img: {:s})'.format(imageid)
            ax1.set_title(t)

        if ax is None and not plt.isinteractive():  # see ImageSet.plot()
            return fig


    def apply(self, image, imageid=None, crop=False, ignore_background=True):
        """ Apply this RegionSet to a specified image.

        Returns a list of the image arrays "cut out" by each region mask, with
        non-selected image areas in black. If regionset is not global, _imageid_ needs
        to be specified!

        Args:
            image (ndarray): image array to be segmented.
            imageid (str): valid imageid (to select image-specific regions if not a global regionset)
            crop (bool): if True, return image cropped to bounding box of selected area
            ignore_background (bool): if True, ignore auto-generated background region

        Returns:
            If crop=False, a list of ndarrays of same size as image, with non-selected areas
            zeroed. Else a list of image patch arrays cropped to bounding box size.
        """
        slices = []
        apply_regions = self._select_region(imageid)
        apply_labels = self._select_labels(imageid)

        for idx, region in enumerate(apply_regions):
            if ignore_background and apply_labels[idx] == '__BG__':
                continue
            mask = (region == True)
            out = np.zeros(image.shape)
            out[mask] = image[mask]

            if crop:
                a = np.argwhere(out)
                if a.shape[0] > 0:
                    (ul_x, ul_y) = a.min(0)[0:2]
                    (br_x, br_y) = a.max(0)[0:2]
                    out = out[ul_x:br_x+1, ul_y:br_y+1]
            slices.append(out)

        return slices


    def export_patches(self, image, imageid=None, crop=True, image_format='png',
                       rescale=False, ignore_background=True):
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
            ignore_background (bool): if True, ignore auto-generated background region

        """
        apply_regions = self._select_region(imageid)
        apply_labels = self._select_labels(imageid)
        imstr = '{:s}_{:s}.{:s}'

        for idx, region in enumerate(apply_regions):
            if ignore_background and apply_labels[idx] == '__BG__':
                continue

            mask = (region == True)
            out = np.zeros(image.shape)
            out[mask] = image[mask]

            if crop:
                a = np.argwhere(out)
                if a.shape[0] > 0:
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
            rimg.save(imstr.format(str(imageid), str(apply_labels[idx]), image_format), image_format)


    def export_patches_from_set(self, imageset, crop=True, image_format='png', rescale=False, ignore_background=True):
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
            ignore_background (bool): if True, ignore auto-generated background region

        """
        if not isinstance(imageset, ImageSet):
            raise TypeError('First argument must be an ImageSet! To slice a single image, use export_patches().')

        for cimg in imageset.imageids:
            if not self.is_global and cimg not in self.imageids:
                print('Warning: RegionSet contains image-specific regions, but no regions available for {:s}. Skipped.'.format(cimg))
            else:
                self.export_patches(imageset[cimg], imageid=cimg, crop=crop, image_format=image_format,
                                    rescale=rescale, ignore_background=ignore_background)
            

    def fixated(self, fixations, var='fixated', imageid=None, exclude_first=False, exclude_last=False):
        """ Returns visited / fixated regions using data from a Fixations object.

        Args:
            fixations (Fixations/DataFrame): fixation data to test against regions
            var (str): type of fixation mapping variable to calculate (default: 'fixated'):
                'fixated': fixation status: 0 - region was not fixated, 1 - fixated (default)
                'count': total number of fixations on each region
                'fixid': fixation ID (from input dataset) for first fixation in each region
            imageid (str): imageid (to select image-specific regions if not a global regionset)
            exclude_first (bool): if True, first fixated region will always be returned as NaN
            exclude_last (str): controls how to deal with regions receiving the last image fixation:
                'never' or False: do not handle the last fixation specially
                'always' or True: drop the entire region if it received the last fixation at any time
                'pass': exclude viewing pass (one or multiple fixations) that received the last fixation

        Returns:
            1D ndarray (float) containing number of fixations per region (if count=True) 
            or the values 0.0 (region was not fixated) or 1.0 (region was fixated)
        """
        if type(exclude_last) == bool:
            if exclude_last:
                exclude_last = 'always'
            elif not exclude_last:
                exclude_last = 'never'

        apply_regions = self._select_region(imageid)
        vis = np.zeros(apply_regions.shape[0], dtype=float)

        # Drop out-of-bounds fixations
        fix = fixations.data[(fixations.data[fixations._xpx] >= 0) & 
                             (fixations.data[fixations._xpx] < self.size[0]) & 
                             (fixations.data[fixations._ypx] >= 0) & 
                             (fixations.data[fixations._ypx] < self.size[1])]
    
        if len(fix) > 0:
            if exclude_first:
                first_fix = fixations.data[fixations.data[fixations._fixid] == min(fixations.data[fixations._fixid])]
                if len(first_fix) > 1:
                    print('Warning: you have requested to drop the first fixated region, but more than one ' +
                          'location ({:d}) matches the lowest fixation ID! Either your fixation ' .format(len(first_fix)) +
                          'IDs are not unique or the passed dataset contains data from multiple images or conditions.')
            if exclude_last != 'never':
                last_fix = fixations.data[fixations.data[fixations._fixid] == max(fixations.data[fixations._fixid])]
                if len(last_fix) > 1:
                    print('Warning: you have requested to drop the last fixated region, but more than one ' +
                          'location ({:d}) matches the highest fixation ID! Either your fixation ' .format(len(last_fix)) +
                          'IDs are not unique or the passed dataset contains data from multiple images or conditions.')

            for (idx, roi) in enumerate(apply_regions):
                if exclude_first:
                    try:
                        is_first = roi[first_fix[fixations._ypx], first_fix[fixations._xpx]]
                        if isinstance(is_first, np.ndarray) and np.any(is_first):
                            vis[idx] = np.nan
                            continue
                        elif is_first:
                            vis[idx] = np.nan
                            continue
                    except IndexError:
                        pass # last fixation is out of bounds for image!

                if exclude_last == 'always':
                    try:
                        is_last = roi[last_fix[fixations._ypx], last_fix[fixations._xpx]]
                        if isinstance(is_last, np.ndarray) and np.any(is_last):
                            vis[idx] = np.nan
                            continue
                        elif is_last:
                            vis[idx] = np.nan
                            continue
                    except IndexError:
                        pass # last fixation is out of bounds for image!

                fv = roi[fix[fixations._ypx], fix[fixations._xpx]]
                if np.any(fv):
                    rfix = fix[fv] # All fixations on region
                    if fixations.has_times:
                        # If fixation data has timing information, ensure to drop fixations
                        # that began before the current fixation report
                        bystart = rfix[rfix[fixations._fixstart] >= 0].sort_values(fixations._fixid)
                    else:
                        bystart = rfix.sort_values(fixations._fixid)

                    if len(bystart) > 0:
                        # Find viewing passes (sets of in-region fixations without leaving region)
                        idxvalid = np.ones(bystart.shape[0], dtype=np.bool) # fix indices to keep
                        idxdiff = bystart[fixations._fixid].diff().reset_index(drop=True)
                        pass_onsets = idxdiff.index.values[(idxdiff > 1)].tolist()
                        num_refix = len(pass_onsets)
                        num_passes = num_refix + 1
                        if len(pass_onsets) >= 1:
                            end_first_pass = pass_onsets[0]
                        else:
                            end_first_pass = bystart.shape[0]

                        # If requested, remove pass containing the last fixation
                        if exclude_last == 'pass':
                            passes = [0,] + pass_onsets + [len(bystart)+1,]
                            for pidx in range(0, len(passes)-1):
                                passfix = bystart.iloc[passes[pidx]:passes[pidx+1], :]
                                if last_fix.index.values[0] in passfix.index:
                                    # Exclude this and all following passes. Note that no later passes
                                    # should exist unless there is an index error in the fixation data!
                                    idxvalid[passes[pidx]:] = False
                                    break

                            if np.all(idxvalid == False):
                                # If no valid fixations remain, drop the whole region (NA)
                                vis[idx] = np.nan
                                continue
                            else:
                                # Keep only valid fixations for fixation count measures
                                bystart = bystart.loc[idxvalid, :]
                                num_refix = np.sum(idxvalid[pass_onsets] == True)
                                num_passes = num_refix + 1

                        # Calculate fixation status measures
                        if var == 'count':
                            # Number of fixations in region
                            vis[idx] = bystart.shape[0]

                        elif var == 'fixated':
                            # Binary coding of fixation status
                            vis[idx] = (bystart.shape[0] >= 1.0)

                        elif var == 'fixid':
                            # Return first valid fixation ID in region
                            if bystart.shape[0] >= 1.0:
                                vis[idx] = bystart.loc[bystart.index[0], fixations._fixid]
                            else:
                                vis[idx] = np.nan

                        elif var == 'passes':
                            # Total number of fixation passes
                            vis[idx] = num_passes

                        elif var == 'refix':
                            # Total number of fixation passes
                            vis[idx] = num_refix

                else:
                    # No fixations in region -> fixID should be NA
                    if var == 'fixid':
                        vis[idx] = np.nan

        return vis


    def fixtimes(self, fixations, var='total', imageid=None, exclude_first=False, exclude_last=False):
        """ Returns fixation-timing based variable for each region. Default is total viewing time.

        Args:
            fixations (Fixations/DataFrame): fixation data to test against regions
            var (str): type of fixation time variable to calculate (default: 'total'):
                'total': total fixation time for each region
                'gaze': gaze duration, i.e. total fixation time in first pass
                'first': first fixation duration per region
                'single': fixation duration if region was fixated exactly once
                'tofirst': start time of the first fixation on each region
            imageid (str): imageid (to select image-specific regions if not a global regionset)
            exclude_first (bool): if True, first fixated region will always be returned as NaN
            exclude_last (str): controls how to deal with regions receiving the last image fixation:
                'never' or False: do not handle the last fixation specially
                'always' or True: drop the entire region if it received the last fixation at any time
                'pass': exclude viewing pass (one or multiple fixations) that received the last fixation

        Returns:
            1D ndarray (float) containing fixation time based dependent variable for each region.
            Regions that were never fixated according to criteria will be returned as NaN.
        """
        if var not in ['total', 'gaze', 'first', 'single', 'tofirst']:
            raise ValueError('Unknown fixation time variable specified: {:s}'.format(var))

        if not fixations.has_times:
            raise AttributeError('Trying to extract a time-based DV from a dataset without fixation timing information! Specify fixstart=/fixend= when loading fixation data!')

        if type(exclude_last) == bool:
            if exclude_last:
                exclude_last = 'always'
            elif not exclude_last:
                exclude_last = 'never'

        apply_regions = self._select_region(imageid)
        ft = np.ones(apply_regions.shape[0], dtype=float) * np.nan

        # Drop out-of-bounds fixations
        fix = fixations.data[(fixations.data[fixations._xpx] >= 0) &
                             (fixations.data[fixations._xpx] < self.size[0]) &
                             (fixations.data[fixations._ypx] >= 0) &
                             (fixations.data[fixations._ypx] < self.size[1])]

        if len(fix) > 0:
            if exclude_first:
                first_fix = fixations.data[fixations.data[fixations._fixid] == min(fixations.data[fixations._fixid])]
                if len(first_fix) > 1:
                    print('Warning: you have requested to drop the first fixated region, but more than one ' +
                          'location ({:d}) matches the lowest fixation ID! Either your fixation ' .format(len(first_fix)) +
                          'IDs are not unique or the passed dataset contains data from multiple images or conditions.')

            if exclude_last != 'never':
                last_fix = fixations.data[fixations.data[fixations._fixid] == max(fixations.data[fixations._fixid])]
                if len(last_fix) > 1:
                    print('Warning: you have requested to drop the last fixated region, but more than one ' +
                          'location ({:d}) matches the highest fixation ID! Either your fixation ' .format(len(last_fix)) +
                          'IDs are not unique or the passed dataset contains data from multiple images or conditions.')

            for (idx, roi) in enumerate(apply_regions):
                if exclude_first:
                    try:
                        is_first = roi[first_fix[fixations._ypx], first_fix[fixations._xpx]]
                        if isinstance(is_first, np.ndarray) and np.any(is_first):
                            ft[idx] = np.nan
                            continue
                        elif is_first:
                            ft[idx] = np.nan
                            continue
                    except IndexError:
                        pass # first fixation is out of bounds for image!

                if exclude_last == 'always':
                    # If this region has the last fixation, drop it here (NaN) and move on
                    try:
                        is_last = roi[last_fix[fixations._ypx], last_fix[fixations._xpx]]
                        if isinstance(is_last, np.ndarray) and np.any(is_last):
                            ft[idx] = np.nan
                            continue
                        elif is_last:
                            ft[idx] = np.nan
                            continue
                    except IndexError:
                        pass # last fixation is out of bounds for image!

                fidx = roi[fix[fixations._ypx], fix[fixations._xpx]]
                if np.any(fidx):
                    rfix = fix[fidx]    # all fixations in this region
                    bystart = rfix[rfix[fixations._fixstart] >= 0].sort_values(fixations._fixid)

                    if len(bystart) > 0:
                        # Find viewing passes (sets of in-region fixations without leaving region)
                        idxvalid = np.ones(bystart.shape[0], dtype=np.bool) # fix indices to keep
                        idxdiff = bystart[fixations._fixid].diff().reset_index(drop=True)
                        pass_onsets = idxdiff.index.values[(idxdiff > 1)].tolist()
                        num_refix = len(pass_onsets)
                        num_passes = num_refix + 1
                        if len(pass_onsets) >= 1:
                            end_first_pass = pass_onsets[0]
                        else:
                            end_first_pass = bystart.shape[0]

                        # If requested, remove pass containing the last fixation
                        if exclude_last == 'pass':
                            passes = [0,] + pass_onsets + [len(bystart)+1,]
                            for pidx in range(0, len(passes)-1):
                                passfix = bystart.iloc[passes[pidx]:passes[pidx+1], :]
                                if last_fix.index.values[0] in passfix.index:
                                    # Exclude this and all following passes. Note that no later passes
                                    # should exist unless there is an index error in the fixation data!
                                    idxvalid[passes[pidx]:] = False
                                    break

                            if np.all(idxvalid == False):
                                # If no valid fixations remain, drop the whole region (NA)
                                ft[idx] = np.nan
                                continue
                            else:
                                # Keep only valid fixations for fixation count measures
                                bystart = bystart.loc[idxvalid, :]
                                num_refix = np.sum(idxvalid[pass_onsets] == True)
                                num_passes = num_refix + 1

                        # Calculate fixation timing measures
                        if var == 'gaze':
                            # Gaze duration: total viewing time of first pass only
                            ft[idx] = sum(bystart.loc[bystart.index[0:end_first_pass], fixations._fixdur])

                        elif var == 'first':
                            # First fixation duration
                            ft[idx] = bystart.loc[bystart.index[0], fixations._fixdur]

                        elif var == 'single':
                            # Single fixation duration (=first fixation duration if not refixated in first pass)
                            ft[idx] = bystart.loc[bystart.index[0], fixations._fixdur]

                            # If refixated on first pass, set to NaN instead
                            if end_first_pass > 1:
                                ft[idx] = np.nan

                        elif var == 'tofirst':
                            # Time until first fixation / first fixation onset
                            ft[idx] = bystart.loc[bystart.index[0], fixations._fixstart]

                        elif var == 'total':
                            # Total viewing time of valid fixations
                            ft[idx] = sum(bystart.loc[:, fixations._fixdur])

        return ft



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
        RegionSet.__init__(self, size=size, regions=regions, label=label, region_labels=region_labels,
                           add_background=False) # GridRegionSets are exhaustive, so the 'background' is empty.

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



class BBoxRegionSet(RegionSet):
    """ RegionSet based on rectangular bounding boxes.

    Attributes:
        cells (list): list of bounding box tuples for each cell,
            each formatted as (left, top, right, bottom)
        label (string): optional label to distinguish between RegionSets
        from_file (string): filename in case regions were loaded from file
        padding (tuple): padding in pixels as ('left', 'top', 'right', 'bottom')
    """

    def __init__(self, size, bounding_boxes, label=None, region_labels=None, sep='\t',
                 imageid='imageid', regionid='regionid', bbox_cols=('x1', 'y1', 'x2', 'y2'),
                 padding=0, add_background=False, coord_format=None):
        """ Create new BBoxRegionSet

        Args:
            size (tuple):   image dimensions, specified as (width, height).
            bounding_boxes: one of the following:
                name of a text/CSV file with columns ([imageid], [regionid], x1, y1, x2, y2)
                list of 4-tuples OR 2D ndarray with columns (x1, y1, x2, y2) for global bboxes
            region_labels (str): list of optional region labels if bounding_boxes is a global array/list
            imageid (str): name of imageid column in input file (if not present, bboxes will be treated as global)
            regionid (str): name of regionid column in input file
            sep (str): separator to use when reading files
            bbox_cols: tuple of column names for ('left', 'top', 'right', 'bottom')
            padding (int): optional bbox padding in pixels as ('left', 'top', 'right', 'bottom'),
                or a single integer to specify equal padding on all sides
            add_background (bool): if True, this creates a special region to capture all
                fixations that don't fall on an explicit region ("background" fixations)
            coord_format (str): Defines how input x and y coordinates are interpreted:
                'oneindexed': coordinates start at 1, e.g. 1..100 for a 100px box
                'zeroindexed': coordinates start at 0, e.g. 0..99 for a 100px box
                'apple': coordinates start at 0, but end at <size>, e.g. 0..100 for a 100px box,
                         in this convention, the pixels sit "between" coordinate values
        """
        self.input_file = None
        self.input_df = None

        self._imageid = imageid
        self._regionid = regionid
        self._cols = bbox_cols

        if coord_format is None:
            err = 'No coordinate format specified! Please provide coord_format argument:\n'
            err += '"oneindexed": coordinates start at 1, e.g. 1..100 for a 100px box\n'
            err += '"zeroindexed": coordinates start at 0, e.g. 0..99 for a 100px box\n'
            err += '"apple": coordinates start at 0, but end at <size>, e.g. 0..100 for a 100px box.'
            raise ValueError(err)

        if type(padding) == int:
            self.padding = (padding,) * 4
        else:
            self.padding = padding

        if isinstance(bounding_boxes, DataFrame):
            # Passed a DataFrame
            bbox = bounding_boxes

        elif type(bounding_boxes) == str:
            # Passed a file name
            try:
                bbox = read_table(bounding_boxes, sep=sep)
                self.input_file = bounding_boxes

            except:
                raise ValueError('String argument supplied to BBoxRegionSet, but not a valid CSV file!')

        else:
            # Try array type
            try:
                bbox = DataFrame(bounding_boxes, columns=['x1', 'y1', 'x2', 'y2'])
            except:
                raise ValueError('Supplied argument to BBoxRegionSet not in the form (x1, y1, x2, y2)')

        (regions, labels) = self._parse_bbox_df(bbox, size, padding, coord_format=coord_format)
        if region_labels is not None:
            labels = region_labels

        RegionSet.__init__(self, size=size, regions=regions, label=label, region_labels=labels, add_background=add_background)

        self.input_df = bbox


    def _parse_bbox_df(self, df, size, padding, coord_format='oneindexed'):
        """ Parse a DataFrame of bounding boxes into a region dict.

        Args:
            df (DataFrame): DataFrame of bounding box coordinates
            size (tuple): image size as (width, height), to check for out-of-bounds coordinates
            padding (int): padding in pixels as ('left', 'top', 'right', 'bottom'), default (0,0,0,0)
            coord_format (str): Defines how input x and y coordinates are interpreted (see __init__)

        Returns:
            tuple of dicts as (regions, labels), using imageids as keys. Resulting dicts can be
            passed to RegionSet.__init__ directly.
        """
        regions = {}
        labels = {}
        _msize = (size[1], size[0])

        if self._imageid in df.columns:
            # Image-specific bounding boxes

            # Force imageid to strings
            df[self._imageid] = df[self._imageid].astype(str)

            for imid, block in df.groupby(self._imageid):
                N = block.shape[0]
                reg = np.zeros((N,) + _msize, dtype=bool)
                lab = []
                bidx = 0

                for idx,row in block.iterrows():
                    # left, top, right, bottom
                    c = [row[self._cols[0]], row[self._cols[1]], row[self._cols[2]], row[self._cols[3]]]

                    if self._regionid in df.columns:
                        l = row[self._regionid]
                    else:
                        l = str(bidx + 1)

                    # Convert coordinates to Python indices
                    if coord_format.lower() == 'oneindexed':
                        c = [round(c[0]) - 1,   # Correct lower bounds
                             round(c[1]) - 1,
                             round(c[2]),
                             round(c[3])]
                    elif coord_format.lower() == 'zeroindexed':
                        c = [round(c[0]),
                             round(c[1]),
                             round(c[2]) + 1,   # Python slices are end-excluding,
                             round(c[3]) + 1]   # i.e. [a, b] does not include b
                    elif coord_format.lower() == 'apple':
                        c = [round(c[0]),       # Just round, format is already fine
                             round(c[1]),
                             round(c[2]),
                             round(c[3])]

                    if c[0] > size[0] or c[2] > size[0] or c[1] > size[1] or c[3] > size[1]:
                        err = 'At least one coordinate of region {:s}/{:s} exceeds the specified image size!'
                        raise ValueError(err.format(imid, l))

                    if c[2]-c[0] < 1 or c[3]-c[1] < 1:
                        err = 'At least one dimension of {:s}/{:s} has a negative length! Columns specified in the wrong order?'
                        raise ValueError(err.format(imid, l))

                    # Add padding, ensuring padded bboxes are cropped to image size
                    c = [c[0] - self.padding[0], c[1] - self.padding[1], c[2] + self.padding[2], c[3] + self.padding[3]]
                    if c[0] < 0:
                        c[0] = 0
                    if c[1] < 0:
                        c[1] = 0
                    if c[2] > size[0]:
                        c[2] = size[0]
                    if c[3] > size[1]:
                        c[3] = size[1]

                    mask = np.zeros(_msize, dtype=bool)
                    mask[c[1]:c[3], c[0]:c[2]] = True
                    reg[bidx,...] = mask
                    lab.append(l)
                    bidx += 1

                regions[imid] = reg
                labels[imid] = lab

        else:
            # Global bounding boxes
            reg = reg = np.zeros((df.shape[0], ) + _msize, dtype=bool)
            lab = []

            for idx,row in df.iterrows():
                c = (row[self._cols[0]], row[self._cols[1]], row[self._cols[2]], row[self._cols[3]])

                if self._regionid in df.columns:
                    l = row[self._regionid]
                else:
                    l = str(idx + 1)

                # Convert coordinates to Python indices
                if coord_format.lower() == 'oneindexed':
                    c = [round(c[0]) - 1,   # Correct lower bounds
                         round(c[1]) - 1,
                         round(c[2]),
                         round(c[3])]
                elif coord_format.lower() == 'zeroindexed':
                    c = [round(c[0]),
                         round(c[1]),
                         round(c[2]) + 1,   # Python slices are end-excluding,
                         round(c[3]) + 1]   # i.e. [a, b] does not include b
                elif coord_format.lower() == 'apple':
                    c = [round(c[0]),       # Just round, format is already fine
                         round(c[1]),
                         round(c[2]),
                         round(c[3])]

                if c[0] > size[0] or c[2] > size[0] or c[1] > size[1] or c[3] > size[1]:
                    err = 'At least one coordinate of region {:s} exceeds the specified image size!'
                    raise ValueError(err.format(l))

                mask = np.zeros(_msize, dtype=bool)
                mask[c[1]:c[3], c[0]:c[2]] = True
                reg[idx,...] = mask
                lab.append(l)

            regions = {'*': reg}
            labels = {'*': lab}

        return (regions, labels)