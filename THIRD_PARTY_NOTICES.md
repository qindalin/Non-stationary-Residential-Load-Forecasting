# Third-Party Notices

## Time-Series-Library (THUML)

Portions of this repository are derived from
[Time-Series-Library](https://github.com/thuml/Time-Series-Library)
(*A Library for Advanced Deep Time Series Models*), in particular:

- `simulation/layers/` (attention/encoder/embedding layers, trimmed to the
  classes used by the DLinear and PatchTST backbones)
- `simulation/models/DLinear.py`, `simulation/models/PatchTST.py` (adapted and
  extended, e.g. PatchTST exogenous correction heads)
- `simulation/data_provider/` (dataset loading structure, adapted)
- `simulation/utils/tools.py`, `simulation/utils/timefeatures.py`,
  `simulation/utils/masking.py`

Copyright (c) 2021 THUML @ Tsinghua University, licensed under the MIT License:

```
MIT License

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

All other code in this repository is original and released under the
repository's [MIT license](LICENSE).

## Data

The bundled client CSV streams under `simulation/data/`, `pi_client/data/`
and `hardware/sample_data/` derive from the Building Data Genome Project 2
(BDG2):

> Miller, C., Kathirgamanathan, A., Picchetti, B., Arjunan, P., Park, J. Y.,
> Nagy, Z., Raftery, P., Hobson, B. W., Shi, Z., Meggers, F. (2020). The
> building data genome project 2, energy meter data from the ASHRAE great
> energy predictor III competition. *Scientific Data*, 7(1), 1–13.
> Licensed under CC BY 4.0 — https://github.com/buds-lab/the-building-data-genome-project-2
