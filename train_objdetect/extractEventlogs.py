import tensorflow as tf
import glob 
tfevents_list=glob.glob('output_inceptionv2_variant_50k/events.out.tfevents*')
for tfevents in tfevents_list:
  for event in tf.train.summary_iterator(tfevents):
      for value in event.summary.value:
          print("Value tag {}".format(value.tag))
          if value.tag == 'loss':
            print("Loss : {}".format(value.simple_value))