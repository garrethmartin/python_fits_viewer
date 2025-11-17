import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from astropy.io import fits
from astropy.visualization import ImageNormalize, LinearStretch, LogStretch, SqrtStretch, AsinhStretch
from ipywidgets import interact, FloatSlider, Dropdown, Button, HBox, VBox, IntSlider
from IPython.display import clear_output, display
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import os
import datetime
from astropy.stats import SigmaClip
from scipy.stats import norm
from photutils.background import Background2D, MedianBackground

class FitsViewer:
    def __init__(self, image, crop=None, figsize=(8,8), mask_file=None):
        
        self.image_data = image
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
            
class BackgroundInteractive:
    def __init__(self, image, figsize=(12, 6), zoom_size=300, inset_frac=0.35):
        self.image = image
        self.zoom_size = zoom_size       # zoom box side length (pixels)
        self.zoom_centre = None          # (x, y) centre of current zoom
        self.inset_frac = inset_frac     # inset size relative to residual panel
        self.bkg_map = None
        self.resid = None
        self.rect = None
        self.ax_inset = None

        self.fig, self.axs = plt.subplots(1, 2, figsize=figsize)
        self.fig.subplots_adjust(wspace=0)
        self.axs = self.axs.flatten()
        for ax in self.axs:
            ax.set_axis_off()

        # connect click on residual
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def _compute_background(self, box_size=128, filter_size=3, sigma=3.0, maxiters=5):
        sigma_clip = SigmaClip(sigma=sigma, maxiters=maxiters)
        bkg_estimator = MedianBackground()
        bkg = Background2D(
            self.image,
            box_size=box_size,
            filter_size=filter_size,
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            edge_method='pad'
        )
        return bkg.background

    def _plot(self, box_size=128):
        self.last_box_size = box_size
        
        clear_output(wait=True)
        print(f'Running for box={box_size}...')

        self.bkg_map = self._compute_background(box_size)
        self.resid = self.image - self.bkg_map

        # determine residual scaling
        lower = (100.0 - 97.5)/2.0
        upper = 100.0 - lower
        vmin_resid, vmax_resid = np.nanpercentile(self.resid, [lower, upper])

        # background
        self.axs[0].cla()
        self.axs[0].imshow(np.arcsinh(self.bkg_map), origin='lower', cmap='viridis')
        self.axs[0].set_title(f'Background (box={box_size})')
        self.axs[0].set_axis_off()

        # residual
        self.axs[1].cla()
        self.axs[1].imshow(np.arcsinh(self.resid), origin='lower', cmap='Greys_r',
                           vmin=vmin_resid, vmax=vmax_resid)
        self.axs[1].set_title('Residual (click to zoom)')
        self.axs[1].set_axis_off()

        # if a zoom was already selected, restore it
        if self.zoom_centre is not None:
            self._draw_zoom_elements()

        self.fig.canvas.draw_idle()
        print('Done.')

    def onclick(self, event):
        if event.inaxes != self.axs[1]:
            return
        self.zoom_centre = (int(event.xdata), int(event.ydata))
        self._draw_zoom_elements()

    def _draw_zoom_elements(self):
        """Draw zoom rectangle and inset for current zoom_centre."""
        if self.resid is None or self.zoom_centre is None:
            return

        x, y = self.zoom_centre
        s = self.zoom_size // 2
        ny, nx = self.resid.shape
        x0, x1 = max(0, x - s), min(nx, x + s)
        y0, y1 = max(0, y - s), min(ny, y + s)

        sub = self.resid[y0:y1, x0:x1]

        # remove any old rectangle/inset
        if self.rect:
            self.rect.remove()
        if self.ax_inset:
            self.ax_inset.remove()

        # draw rectangle on residual
        self.rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                              edgecolor='red', facecolor='none', lw=2)
        self.axs[1].add_patch(self.rect)

        # compute normalisation for inset
        lower = (100.0 - 97.5)/2.0
        upper = 100.0 - lower
        vmin, vmax = np.nanpercentile(sub, [lower, upper])
        norm = ImageNormalize(sub, vmin=vmin, vmax=vmax, stretch=LogStretch())

        # inset location (top-left)
        self.ax_inset = inset_axes(self.axs[1],
                                   width=f"{int(self.inset_frac*100)}%",
                                   height=f"{int(self.inset_frac*100)}%",
                                   loc='upper left',
                                   borderpad=1)
        self.ax_inset.imshow(sub, origin='lower', cmap='Greys', norm=norm)
        self.ax_inset.set_xticks([])
        self.ax_inset.set_yticks([])

        self.fig.canvas.draw_idle()

    def _update_zoom(self, zoom_size):
        """Callback to update zoom size interactively."""
        self.zoom_size = zoom_size
        if self.zoom_centre is not None:
            self._draw_zoom_elements()

    def interact(self):
        box_slider = IntSlider(value=128, min=32, max=1024, step=32,
                               description='box size', continuous_update=False)
        zoom_slider = IntSlider(value=self.zoom_size, min=100, max=1000, step=100,
                                description='zoom size', continuous_update=True)

        interact(self._plot, box_size=box_slider)
        zoom_slider.observe(lambda change: self._update_zoom(change['new']), names='value')

        display(VBox([zoom_slider]))
        
    def plot_residual_hist(self, tile_size=50, bins=50, clip_sigma=3):
            """
            Plot residual histogram with Gaussian overlay.
            """
            if self.resid is None:
                print("Residual not computed yet. Run _plot() first.")
                return
            
            box_size = getattr(self, 'last_box_size', None)

            residual = self.resid
            img_sky = self.image
            ny, nx = residual.shape
            tile_stds = []

            # Compute standard deviations in tiles
            for y0 in range(0, ny, tile_size):
                for x0 in range(0, nx, tile_size):
                    y1 = min(y0 + tile_size, ny)
                    x1 = min(x0 + tile_size, nx)
                    tile = img_sky[y0:y1, x0:x1]
                    tile_stds.append(np.std(tile))

            # Median sigma across all tiles (for Gaussian overlay)
            sigma_bg = np.median(tile_stds)

            # Flatten residuals
            data = residual.ravel()

            # Sigma clipping as before
            if clip_sigma is not None:
                mean = np.mean(data)
                std = np.std(data)
                mask = (data > mean - clip_sigma*std) & (data < mean + clip_sigma*std)
                clipped_data = data[mask]
            else:
                clipped_data = data
                mean = np.mean(data)
                std = np.std(data)

            # Histogram plot
            plt.figure(figsize=(5,4))
            counts, bins_edges, _ = plt.hist(clipped_data, bins=bins, density=True,
                                             color='gray', edgecolor='black', alpha=0.7, label='Residuals')

            # Gaussian overlay centered on 0
            x = np.linspace(bins_edges[0], bins_edges[-1], 200)
            plt.plot(x, norm.pdf(x, 0, sigma_bg), 'r--', lw=2, label='Expected Gaussian noise')

            plt.xlabel("Residual value")
            plt.ylabel("Normalized count")
            plt.title(rf'Residuals (box={box_size})')
            plt.legend(loc='lower left')
            plt.grid(True, alpha=0.3)
            plt.show()
