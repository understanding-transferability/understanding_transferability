"""Simple example on how to log scalars and images to tensorboard without tensor ops.
License: Copyleft
"""
__author__ = "Michael Gygli"

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorlayer as tl
import six
StringIO = six.StringIO

import numpy as np
from wheel import to_gray_np, to_rgb_np
import os

class Logger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir, clear=False):
        """
        Creates a summary writer logging to log_dir.

        :param log_dir: directory to keep the log. will create it if the directory doesn't exist
        :param clear: if the directory is not empty, whether to clear the directory
        """
        if clear:
            os.system('rm %s -r'%log_dir)
        tl.files.exists_or_mkdir(log_dir)
        self.writer = tf.summary.FileWriter(log_dir)
        self.step = 0
        self.log_dir = log_dir

    def log_scalar(self, tag, value, step=None):
        """
        Log a scalar variable.

        :param tag: Name of the scalar
        :param value: value of the scalar
        :param int step: training iteration (if not provided, use Logger.step instead)
            **it's recommended that Logger keep track with the step value**
        """
        if not step:
            step = self.step
        
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_images(self, tag, images, step=None):
        """
        Logs a list of images.

        :param tag:
        :param images: can be 3-D tensor [N, H, W](gray image) or 4-D tensor [N, H, W, C],
            C = 1 or C = 3 or C = 4, C = 1 is gray image
        :param int step: training iteration (if not provided, use Logger.step instead)
            **it's recommended that Logger keep track with the step value**
        """

        if not step:
            step = self.step
        
        im_summaries = []
        for nr, img in enumerate(images):
            # Write the image to a string
            s = StringIO()
            
            if len(img.shape) == 2:
                img = np.expand_dims(img, axis=-1)
            
            if img.shape[-1] == 1:
                img = np.tile(img, [1, 1, 3])
            img = to_rgb_np(img)
            plt.imsave(s, img, format='png')

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_histogram(self, tag, values, step=None, bins=1000):
        """
        Logs the histogram of a list/vector of values.

        :param tag:
        :param values:
        :param int step: training iteration (if not provided, use Logger.step instead)
            **it's recommended that Logger keep track with the step value**
        :param bins:
        """
        if not step:
            step = self.step
        
        # Convert to a numpy array
        values = np.array(values)
        
        # Create histogram using numpy        
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()

    def log_bar(self, tag, values, xs=None, step=None):
        """
        log bar chart (hack tensorflow to draw a histogram that looks like a bar chart)
        :param tag:
        :param values:
        :param xs:
        :param step:
        :return:
        """
        if not step:
            step = self.step

        values = np.asarray(values).flatten()
        if not xs:
            axises = list(range(len(values)))
        else:
            axises = xs

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(min(axises))
        hist.max = float(max(axises))
        hist.num = sum(values)
        hist.sum = sum([y * x for (x, y) in zip(axises, values)])
        hist.sum_squares = sum([y * (x ** 2) for (x, y) in zip(axises, values)])

        for edge in axises:
            hist.bucket_limit.append(edge - 1e-10)
            hist.bucket_limit.append(edge + 1e-10)
        for c in values:
            hist.bucket.append(0)
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, self.step)
        self.writer.flush()