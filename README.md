# Xavier 
People Tracking System

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

## Testing

### test.py python command
this python file can be used to perform the evaluation of the precision and recall of the algorithm combination used

```bash
$ python -m test -h

usage: test.py [-h] --input INPUT --label_dir LABEL_DIR
               [--output_dir OUTPUT_DIR] [--detector DETECTOR] [--conf CONF]
               [--tracker] [--skip SKIP] [--show] [--save]

Run "test" to get precision and recall

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         file to detect and track
  --label_dir LABEL_DIR
                        directory where there are saved the label of the file
  --output_dir OUTPUT_DIR
                        path for the output file
  --detector DETECTOR   detector to use [yolov3, yolov3Conf, hog]
  --conf CONF           configuration dir for yolov3Conf
  --tracker             combination of detection and tracker to detection task
  --skip SKIP           number of frame to skip after detection
  --show                if show the video in output
  --save                if you want show the evaluation parameters
```

### test_AP.py python command
this python file can be used to compute the Average Precision of the algorithm combination used

```bash
$ python -m test_AP -h

usage: test_AP.py [-h] --input INPUT --label_dir LABEL_DIR
                  [--detector DETECTOR] [--conf CONF] [--tracker]
                  [--skip SKIP] [--th TH] [--show]

Run "test_AP" to get Average Precision

optional arguments:
  -h, --help            show this help message and exit
  --input INPUT         file to detect and track
  --label_dir LABEL_DIR
                        directory where there are saved the label of the file
  --detector DETECTOR   detector to use [yolov3, yolov3Conf, hog]
  --conf CONF           configuration dir for yolov3Conf
  --tracker             combination of detection and tracker to detection task
  --skip SKIP           number of frame to skip after detection
  --th TH                threshold for IoU to consider True Positive
  --show                if show the video in output
```