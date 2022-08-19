# radarqc

Python utilities for loading and processing HF radar spectra in Cross-Spectrum file format.
See file specification [here](http://support.codar.com/Technicians_Information_Page_for_SeaSondes/Manuals_Documentation_Release_8/File_Formats/File_Cross_Spectra_V6.pdf).

## Overview
This repository provides a python package capable of:
  - Loading Cross-Spectrum files as Python objects containing headers and antenna spectra.
  - Serializing Python object representation as Cross-Spectrum files.
  - Preprocessing antenna spectra to calculate gain and deal with outliers.
  - Filtering spectra to reduce the effects of background noise on wave velocity calculation.

## Installation
```bash
pip3 install radarqc
```

## Example Usage
The radar used to generate cross-spectrum data can sometimes detect outliers.  This is indicated by 
negative signal values in the data.  This example loads a file using the `Abs` method to ignore the outliers,
computes the log-power (dBW), then writes the result back into a file.

```python3
from radarqc import csfile
from radarqc.processing import Abs, CompositeProcessor, GainCalculator

def example():
    reference_gain_db = 4.2
    path = "example.cs"
    preprocess = CompositeProcessor(
        Abs(), GainCalculator(reference=reference_gain_db)
    )
    
    # Read binary file into 'CSFile' object.
    # Spectrum data will be processed to compute gain.
    with open(path, "rb") as f:
        cs = csfile.load(f, preprocess)
    
    # Write processed file back into original format on disk.
    with open(path, "wb") as f:
        csfile.dump(cs, f)
```

The loaded `CSFile` object can be used to access file metadata via the `header` attribute, as well as various attributes for accessing data from individual antenna and cross-antenna spectra with a `numpy.ndarray` data type.

The `radarqc` package also supports conversion from `CSFile` objects to [xarray.Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html) objects.  This interoperability allows for easy conversion from cross-spectrum files to [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) files.

```python3
import matplotlib.pylot as plt

from radarqc import csfile
from radarqc.processing import calculate_gain

def example():
    with open(input_path, "rb") as f:
        # Load cross-spectrum file as xarray dataset.
        ds = csfile.load(f).to_xarray()

    # Print human-readable representation of xarray dataset.
    print(ds)

    # Plot antenna1 log-power (gain) in dB.
    calculate_gain(ds.antenna1).plot()

    plt.tight_layout()
    plt.show()

    # Save xarray dataset as netcdf file.
    ds.to_netcdf(output_path)
```

When converted from `CSFile` objects, `xarray.Dataset` objects will be automagically populated with range and frequency coordinates derived from the cross-spectra file metadata, which enables easy and accurate plotting of individual antenna spectra.  The plot from the above example will look like the following:

<p align="center">
  <img src="https://github.com/jstanco/radarqc/blob/dev/docs/antenna1.jpg?raw=true" />
</p>
