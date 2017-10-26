import numpy as np
import PIL
import PIL.Image
import os
from skimage import color
import sys
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description = "Tensorflow image quantization")
parser.add_argument("input", help="imput image file")
parser.add_argument("output", help="output image file")
parser.add_argument("colors", help="color count", type=int)
parser.add_argument("-d", "--history", help="create an output file every N steps", type=int, metavar="N")
parser.add_argument("-m", "--mask", help="use MASK to generate filenames for files created with --history option, default is '%%05d.png'", default='%05d.png', metavar="MASK")
parser.add_argument("-q", "--quiet", help="do not output progress", action="store_true")
args = parser.parse_args()

def info(text):
    if not args.quiet:
        print(text)

def maximin(x):
    centers = np.array([x[0]])
    while centers.shape[0] < args.colors:
        distances = []
        for i in range(centers.shape[0]):
            distances += [ np.sum(np.square(x-centers[i]),axis=1) ]
        distance_to_closest = np.min(np.array(distances), axis=0)
        index = np.argmax(distance_to_closest)
        centers =  np.vstack([centers, x[index]])
    
    return np.array(centers, dtype=np.float32)

channelCount = 3
image = PIL.Image.open(args.input).convert('RGB')
width, height = image.size

ax = color.rgb2lab(np.asarray(image).reshape([height, width, channelCount]).astype('float32') * 1.0 / 255)
ax = ax.reshape([height * width, channelCount]);
initial_colors = maximin(ax)

input = tf.placeholder(tf.float32, shape=(width*height, channelCount))
images_reshaped = tf.reshape(input, [width*height, 1, channelCount])
images_tiled = tf.tile(images_reshaped, [1,args.colors,1])

colors = tf.Variable( tf.constant(initial_colors*0.9999), name="lab")

colors_reshaped = tf.reshape(colors, [1, args.colors, channelCount])
colors_tiled = tf.tile(colors_reshaped, [width*height,1,1])

difference = tf.reduce_sum(tf.abs(images_tiled - colors_tiled), axis=2)

indexes = tf.argmax(-difference, axis=1)
restored_picture = tf.gather(colors, indexes)

worst_colors = tf.argmax(difference, axis=1)
restored_worst = tf.gather(colors, worst_colors)
difference = restored_picture - input

error_unevenness = tf.reduce_mean(tf.abs(difference - tf.reduce_mean(difference)))
error_best = tf.sqrt(tf.reduce_sum(tf.square(difference), axis=1))
error_worst = tf.sqrt(tf.reduce_sum(tf.square(restored_worst - input), axis=1))

loss_early = tf.reduce_mean(error_best) + 0.5 * tf.reduce_mean(error_worst)
loss_late = tf.reduce_mean(error_best)
loss_var = loss_late

train_step_early = tf.train.AdamOptimizer(learning_rate=15).minimize(loss_early)
train_step_later = tf.train.AdamOptimizer(learning_rate=2).minimize(loss_late)
train_step = train_step_later

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def save_picture(name):
    v = sess.run(restored_picture, feed_dict={ input: ax })
    imgdata = v.reshape([height, width, channelCount])

    imgdata=color.lab2rgb(np.array(imgdata, dtype=np.float64))
    imgdata=np.array((imgdata*255).clip(min=0,max=255),dtype=np.int8)

    img = PIL.Image.fromarray(imgdata, 'RGB')
    img.save(name)

class RunningAverage:
    def __init__(self, s):
        self.size = s
        self.list = []

    def add(self, v):
        self.list += [v]
        
        if len(self.list)>self.size:
            self.list.pop(0)
            
    def clear(self):
        self.list = []
    
    def mean(self):
        if len(self.list) == 0:
            return 0
        
        return  sum(self.list) / len(self.list)

phase = 1
runningAverageSize = 30
longAverage = RunningAverage(runningAverageSize)
shortAverage = RunningAverage(runningAverageSize/2)
stepToStartCheckingImprovement = runningAverageSize
index = 1

def nextPhase():
    global step
    global runningAverageSize
    global phase
    global train_step
    global loss_var
    global stepToStartCheckingImprovement
    global shortAverage
    global longAverage

    shortAverage.clear()
    longAverage.clear()
    
    if phase==0:
        phase=1
        train_step = train_step_later
        loss_var = loss_late
        stepToStartCheckingImprovement = step + runningAverageSize
    elif phase==1:
        phase=0
        train_step = train_step_early
        loss_var = loss_early
        stepToStartCheckingImprovement = step + runningAverageSize
    
    info('switching to phase '+str(phase))

for step in range(10000):
    _, loss = sess.run([train_step, loss_var], feed_dict={ input: ax })
    longAverage.add(loss)
    shortAverage.add(loss)
    improvement = longAverage.mean() - shortAverage.mean();
   
    info('step: '+str(step)+', loss: '+str(loss)+", improvement: "+str(improvement));
    
    if step>stepToStartCheckingImprovement and improvement<0.01 and phase==0:
        nextPhase()
    elif step>stepToStartCheckingImprovement and improvement<0.001 and phase==1:
        indexes_value = sess.run([indexes], feed_dict={ input: ax })
        color_count = np.unique(indexes_value).shape[0]
        if color_count == args.colors:
            info('finished')
            break
        
        info('color count in result: '+str(color_count))
        nextPhase()

    if(args.history > 0 and step % args.history==0):
        save_picture(args.mask % index)
        index+=1

save_picture(args.output);

