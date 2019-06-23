# Xavier 
Detect and Track People

## Requirements

- Python: Python 3.*
- Packages: requirements.txt

## How to Use

```bash
$ cd source

$ virtualenv -p python3 venv

$ source venv/bin/activate

$ pip install -r requirements.txt

```

### detect_to_track.py python command
this python file can be executed by command line in this way:

```bash
$ python -m detect_to_track -h
usage: detect_to_track.py [-h] --input INPUT [--detector DETECTOR]
                          [--tracker TRACKER] [--save] [--label] [--trace]
                          [--top] [--matrix MATRIX]

optional arguments:
  -h, --help           show this help message and exit
  --input INPUT        file to detect and track
  --detector DETECTOR  detector to use [yolov3, hog]
  --tracker TRACKER    tracker to use [open, sort, centroid]
  --show               if show the video in output
  --save               if save the final output
  --label              if save the label
  --trace              if show the trace
  --top                if show the view from above
  --matrix MATRIX      file contain matrix to change camera view
```
