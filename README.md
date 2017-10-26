Tensorflow color quantizaton
============================

```
usage: train.py [-h] [-d N] [-m MASK] [-q] input output colors

Tensorflow image quantization

positional arguments:
  input                 imput image file
  output                output image file
  colors                color count

optional arguments:
  -h, --help            show this help message and exit
  -d N, --history N     create an output file every N steps
  -m MASK, --history-mask MASK
                        use MASK to generate filenames for files created with
                        --history option, default is '%05d.png'
  -q, --quiet           do not output progress
```
