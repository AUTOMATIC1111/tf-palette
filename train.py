import numpy as np
import PIL
import PIL.Image
import os
from skimage import color
import sys

import tensorflow as tf

filename = sys.argv[1]
outputFilename = sys.argv[2]
colorCount = int(sys.argv[3])
saveEveryFilenameMask = sys.argv[4]
saveEvery = int(sys.argv[5])

channelCount = 3
image = PIL.Image.open(filename).convert('RGB')
width, height = image.size

ax = color.rgb2lab(np.asarray(image).reshape([height, width, channelCount]).astype('float32') * 1.0 / 255)
ax = ax.reshape([height * width, channelCount]);

input = tf.placeholder(tf.float32, shape=(width*height, channelCount))
images_reshaped = tf.reshape(input, [width*height, 1, channelCount])
images_tiled = tf.tile(images_reshaped, [1,colorCount,1])

color_l = tf.Variable( tf.random_uniform([colorCount], minval=0, maxval=100), name="l" )
color_a = tf.Variable( tf.random_uniform([colorCount], minval=-128, maxval=128), name="a" )
color_b = tf.Variable( tf.random_uniform([colorCount], minval=-128, maxval=128), name="b" )
colors = tf.stack( [ color_l, color_a, color_b ], axis=1 )

colors_reshaped = tf.reshape(colors, [1, colorCount, channelCount])
colors_tiled = tf.tile(colors_reshaped, [width*height,1,1])

difference = tf.reduce_sum(tf.abs(images_tiled - colors_tiled), axis=2)

indexes = tf.argmax(-difference, axis=1)
restored_picture = tf.gather(colors, indexes)

worst_colors = tf.argmax(difference, axis=1)
restored_worst = tf.gather(colors, worst_colors)

loss_best = tf.sqrt(tf.reduce_sum(tf.square(restored_picture - input), axis=1))
loss_worst = tf.sqrt(tf.reduce_sum(tf.square(restored_worst - input), axis=1))

loss = tf.reduce_sum( loss_best + 0.1 * loss_worst )
later_loss = tf.reduce_sum( loss_best )

train_step_initial = tf.train.GradientDescentOptimizer (learning_rate=0.000015).minimize(loss)
train_step_later = tf.train.GradientDescentOptimizer (learning_rate=0.000015).minimize(later_loss)
train_step = train_step_initial

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def save_picture(name):
    v = sess.run(restored_picture, feed_dict={ input: ax })
    imgdata = v.reshape([height, width, channelCount])

    imgdata=color.lab2rgb(np.array(imgdata, dtype=np.float64))
    imgdata=np.array((imgdata*255).clip(min=0,max=255),dtype=np.int8)

    img = PIL.Image.fromarray(imgdata, 'RGB')
    img.save(name)

for step in range(10000):
    if step==2000:
        train_step=train_step_later
    
    _, loss_val = sess.run([train_step, loss], feed_dict={ input: ax })
    
    print('step: '+str(step)+', loss: '+str(loss_val))
    
    if(saveEvery > 0 and step%saveEvery==0):
        save_picture(saveEveryFilenameMask % step);
        
        last_loss=loss_val

save_picture(outputFilename);

