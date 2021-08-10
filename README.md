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
computes the relative gain, then writes the result back into a file.

```python3
from radarqc import csfile
from radarqc.processing import Abs, CompositeProcessor, GainCalculator

def example():
    reference_dbm = 34.2
    path = "example.cs"
    preprocess = CompositeProcessor(
        Abs(), GainCalculator(reference=reference_dbm)
    )
    
    # Read binary file into 'CSFile' object.
    # Spectrum data will be processed to compute gain.
    with open(path, "rb") as f:
        cs = csfile.load(f, preprocess)
    
    # Write processed file back into original format on disk.
    with open(path, "wb") as f:
        csfile.dump(cs, f)
```

The loaded `CSFile` object can be used to access file metadata via the `header` attribute,
as well as various attributes for accessing data from individual antenna and cross-antenna spectra
with a `numpy.ndarray` data type.

