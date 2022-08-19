# radarqc

Python utilities for loading and processing HF radar spectra in Cross-Spectrum file format.
See file specification [here](http://support.codar.com/Technicians_Information_Page_for_SeaSondes/Manuals_Documentation_Release_8/File_Formats/File_Cross_Spectra_V6.pdf).

## Overview
This repository provides a python package capable of:
  - Loading Cross-Spectrum files as Python objects containing headers and antenna spectra.
  - Serializing Python object representation as Cross-Spectrum files.
  - Extensible preprocessing interface for antenna spectra.
  - Filtering spectra to reduce the effects of background noise on wave velocity calculation.
  - Conversion to `xarray.Dataset` objects that may be converted to NetCDF files.

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

## Future Work

### Better handling of complex arrays when converting to xarray.Dataset

One constraint of NetCDF is lack of support for complex-valued data.  This poses no problem for self-spectra, as they contain no imaginary component.  However, this makes storage of cross-spectra tricky.  In practice, this can be *almost* solved by interpreting `np.complex64` data items as 2 contiguous `np.float32` items.  This allows for serialization of cross-spectra into NetCDF format without any difficulties.  However, this means that in order to process cross-spectra arrays as complex (which is necessary for many signal-processing algorithms), one must first call `x.view(dtype=np.complex64)`.

### Full support for variable metadata

Currently, the Cross-Spectra file format version 6 supports the addition of variable key-value based metadata.  The `radarqc.csfile` module *does* contain methods for handling all possible keys.  However, it does not interpret these fields when converting to xarray, preferring instead to create a global dataset attribute that maps to a the key-value data as a `JSON` string.  However, it does not enable first-class support for these variable key/values as xarray dataset attributes.

### Exploring Options for HF Radar Data coordinate conventions.

In order to maximize re-usability, it would be beneficial for NetCDF files created by the `radarqc.csfile` module to be fully interoperable with existing databases for storing range-dependent Doppler spectra.  This would require research into what databases are commonly used in this field, as well as an exploration into their common conventions.
