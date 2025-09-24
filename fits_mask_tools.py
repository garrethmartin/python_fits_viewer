import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, LogStretch, SqrtStretch, AsinhStretch
from ipywidgets import interact, FloatSlider, Dropdown, Button, HBox
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import os
import datetime

class FitsViewer:
    def __init__(self, filename, hdu_index=0, crop=None, figsize=(8,8), mask_file=None):
        # load fits file
        hdu = fits.open(filename)
        self.image_data = hdu[hdu_index].data
        hdu.close()
        
        if crop:
            self.image_data = self.image_data[crop[0]:crop[1], crop[2]:crop[3]]

        # load mask if provided
        self.mask = None
        self.display_data = self.image_data.copy()
        if mask_file is not None:
            mask_hdu = fits.open(mask_file)
            self.mask = mask_hdu[0].data.astype(bool)
            mask_hdu.close()
            self.display_data = np.ma.masked_where(~self.mask, self.image_data)
            
            # crop to masked region
            rows, cols = np.where(self.mask)
            rmin, rmax = rows.min(), rows.max()+1
            cmin, cmax = cols.min(), cols.max()+1
            self.display_data = self.display_data[rmin:rmax, cmin:cmax]
            self.image_data = self.image_data[rmin:rmax, cmin:cmax]
            if self.mask is not None:
                self.mask = self.mask[rmin:rmax, cmin:cmax]

        # create figure and axes
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_axis_off()
        divider = make_axes_locatable(self.ax)
        self.cax = divider.append_axes("right", size="3%", pad=0.05)
        
        # initial display
        self.scaling = 99.0
        self._compute_vmin_vmax()
        self.im = self.ax.imshow(self.display_data, origin='lower', cmap='gray',
                                 norm=ImageNormalize(self.image_data, vmin=self.vmin, vmax=self.vmax, stretch=LinearStretch()),
                                 interpolation='nearest')
        self.cbar = self.fig.colorbar(self.im, cax=self.cax)
        
        # launch interactive widgets
        self._create_widgets()
    
    def _compute_vmin_vmax(self):
        # always compute from the cropped original data for consistent scaling
        lower = (100.0 - self.scaling)/2.0
        upper = 100.0 - lower
        self.vmin, self.vmax = np.nanpercentile(self.image_data, [lower, upper])

    @property
    def masked_data_cropped(self):
        if self.mask is not None:
            rows, cols = np.where(self.mask)
            rmin, rmax = rows.min(), rows.max()+1
            cmin, cmax = cols.min(), cols.max()+1
            return np.ma.masked_where(~self.mask[rmin:rmax, cmin:cmax], self.image_data[rmin:rmax, cmin:cmax])
        else:
            return self.image_data
    
    def _create_widgets(self):
        # create interactive widgets
        self.stretch_widget = Dropdown(options=['linear','log','sqrt','asinh'], value='linear', description='stretch')
        self.contrast_slider = FloatSlider(value=1.0, min=0.1, max=5.0, step=0.05, description='contrast')
        self.white_slider = FloatSlider(value=1.0, min=0.0, max=1.0, step=0.01, description='white')
        self.scaling_slider = FloatSlider(value=99.0, min=90.0, max=100.0, step=0.1, description='scaling (%)')
        
        # snapshot button
        self.snapshot_button = Button(description="take snapshot", button_style='success')
        self.snapshot_button.on_click(self._take_snapshot)
        
        # display interactive widgets
        interact(self.update_image,
                 stretch_type=self.stretch_widget,
                 contrast=self.contrast_slider,
                 white_frac=self.white_slider,
                 scaling=self.scaling_slider)
        display(self.snapshot_button)
    
    def _take_snapshot(self, b):
        # save current image display as png
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fits_snapshot_{timestamp}.png"
        img = self.im.get_array()
        plt.imsave(filename, np.array(img), cmap=self.im.get_cmap(), vmin=self.im.get_clim()[0], vmax=self.im.get_clim()[1])
        print(f"snapshot saved as {filename}")

    def update_image(self, stretch_type='linear', contrast=1.0, white_frac=1.0, scaling=99.0):
        # update display with stretch, contrast, white fraction, and scaling
        self.scaling = scaling
        self._compute_vmin_vmax()
        vmax_adj = self.vmin + white_frac * (self.vmax - self.vmin)
        vcenter = 0.5 * (self.vmin + vmax_adj)
        vhalf_range = 0.5 * (vmax_adj - self.vmin) / contrast
        vmin_norm = vcenter - vhalf_range
        vmax_norm = vcenter + vhalf_range
        
        # choose stretch
        stretch_map = {'linear': LinearStretch(), 'log': LogStretch(), 'sqrt': SqrtStretch(), 'asinh': AsinhStretch()}
        stretch = stretch_map.get(stretch_type, LinearStretch())
        
        # update normalization
        norm = ImageNormalize(self.image_data, vmin=vmin_norm, vmax=vmax_norm, stretch=stretch)
        self.im.set_norm(norm)
        self.cbar.update_normal(self.im)
        self.fig.canvas.draw_idle()
        
        # store current display settings
        self.stretch_type = stretch_type
        self.contrast = contrast
        self.white_frac = white_frac
            

class MaskPainter:
    def __init__(self, fv, brush_size=5, figsize=(8,8)):
        self.fv = fv
        self.image_data = fv.image_data
        self.brush_size = brush_size

        # get normalization from fits viewer
        self.stretch_type = fv.stretch_type
        self.contrast = fv.contrast
        self.white_frac = fv.white_frac
        self.scaling = fv.scaling
        self.norm = self._compute_norm()

        # create figure
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.ax.set_axis_off()
        self.im = self.ax.imshow(self.image_data, origin='lower', cmap='gray', norm=self.norm, interpolation='nearest')

        # mask overlay
        self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
        self.masks = []
        self.mask_colors = []
        self.overlays = []

        # mouse events
        self.painting = False
        self.cid_press = self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self._on_release)
        self.cid_motion = self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)

        # buttons
        self.next_button = Button(description='next mask', button_style='info')
        self.next_button.on_click(self._next_mask)
        self.save_button = Button(description='save masks', button_style='success')
        self.save_button.on_click(self._save_masks)
        display(HBox([self.next_button, self.save_button]))

        os.makedirs('./masks', exist_ok=True)
        self.fig.canvas.draw_idle()

    def _compute_norm(self):
        # compute image normalization
        self.fv._compute_vmin_vmax()
        vmax_adj = self.fv.vmin + self.white_frac * (self.fv.vmax - self.fv.vmin)
        vcenter = 0.5 * (self.fv.vmin + vmax_adj)
        vhalf_range = 0.5 * (vmax_adj - self.fv.vmin) / self.contrast
        vmin_norm = vcenter - vhalf_range
        vmax_norm = vcenter + vhalf_range
        stretch_map = {'linear': LinearStretch(), 'log': LogStretch(), 'sqrt': SqrtStretch(), 'asinh': AsinhStretch()}
        stretch = stretch_map.get(self.stretch_type, LinearStretch())
        return ImageNormalize(self.image_data, vmin=vmin_norm, vmax=vmax_norm, stretch=stretch)

    def _on_press(self, event):
        # start painting
        if event.inaxes != self.ax:
            return
        self.painting = True
        self._paint(event)

    def _on_release(self, event):
        # stop painting
        self.painting = False

    def _on_motion(self, event):
        # paint on motion
        if self.painting and event.inaxes == self.ax:
            self._paint(event)

    def _paint(self, event):
        # paint circular brush
        x0, y0 = int(round(event.xdata)), int(round(event.ydata))
        yy, xx = np.ogrid[:self.image_data.shape[0], :self.image_data.shape[1]]
        mask_circle = (yy - y0)**2 + (xx - x0)**2 <= self.brush_size**2
        self.current_mask[mask_circle] = 1

        # update overlay
        if hasattr(self, 'current_overlay_im'):
            self.current_overlay_im.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
        else:
            self.current_overlay_im = self.ax.imshow(
                np.ma.masked_where(self.current_mask==0, self.current_mask),
                origin='lower', cmap='Reds', alpha=0.5, interpolation='nearest'
            )
        self.fig.canvas.draw_idle()

    def _next_mask(self, b):
        # save current mask and start new
        if np.any(self.current_mask):
            color = np.random.rand(3,)
            self.mask_colors.append(color)
            self.masks.append(self.current_mask.copy())
            overlay = self.ax.imshow(np.ma.masked_where(self.current_mask==0, self.current_mask),
                                     origin='lower', cmap=mcolors.ListedColormap([color]),
                                     alpha=0.5, interpolation='nearest')
            self.overlays.append(overlay)

        self.current_mask = np.zeros_like(self.image_data, dtype=np.uint8)
        if hasattr(self, 'current_overlay_im'):
            self.current_overlay_im.set_data(np.ma.masked_where(self.current_mask==0, self.current_mask))
        self.fig.canvas.draw_idle()
        print(f"started new mask (total saved: {len(self.masks)})")

    def _save_masks(self, b):
        # save all masks to fits files
        for i, mask in enumerate(self.masks):
            filename = f"./masks/mask_{i:03d}.fits"
            fits.PrimaryHDU(mask.astype(np.uint8)).writeto(filename, overwrite=True)
            print(f"saved {filename}")
