# FITS Viewer and Mask Painter

A Python toolkit for interactive viewing of FITS images and painting masks. Designed for astronomers and data scientists working with low surface brightness (LSB) features.  

## Features

- **Interactive FITS viewing**  
  - Adjustable contrast, stretch (`linear`, `log`, `sqrt`, `asinh`), white fraction, and percentile scaling.  
  - Save snapshots of the current display as PNG.  

- **Mask painting**  
  - Paint multiple masks on the image using a circular brush.  
  - Assign random colours to each mask.  
  - Save masks as FITS files.  
  - Optionally crop and display masked regions only.  

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/garrethmartin/python_fits_viewer.git
cd python_fits_viewer
pip install -r requirements.txt
```
