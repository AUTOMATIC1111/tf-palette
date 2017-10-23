Tensorflow color quantizaton
============================

Run the program as:

```
python train.py <input> <output> <color-count> <filename-with-mask> <N>
```

Every `<N>` optimization steps, the result will be written out to file with name `<filename-with-mask>`. Set `<N>` to 0 to disable.

You must specify all 5 arguments.

For example:

```
python train.py 1477775644004.jpg res.png 4 res%04d.png 100
```
