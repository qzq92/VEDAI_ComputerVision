# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from datetime import datetime
import pytz
from absl import flags
import time
import os
from pathlib import Path
import sys
import logging
#Get the directory of this python file
current_file_dir = os.path.dirname(os.path.realpath("__file__"))
path = Path(current_file_dir)

#Add tensorflow models/research folder for external imports
tf_models = os.path.join(path.parent.parent,"models","research")
slim_dir = os.path.join(path.parent.parent,"models","research","slim")
local_bin = os.path.join("/usr","local","bin")

print("Adding tensorflow models: {}, {}, {} to sys path...".format(tf_models,slim_dir,local_bin))
sys.path.insert(0,tf_models)
sys.path.insert(0,slim_dir)
sys.path.insert(0,local_bin)

import tensorflow as tf
from object_detection import model_hparams
from object_detection import model_lib
import warnings
#Ignore deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import csv
import io

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
class CsvFormatter(logging.Formatter):
    def __init__(self,output_csv):
        super().__init__()
        self.output_csv = open(output_csv, 'w')
        self.output = io.StringIO()
        self.writer = csv.writer(self.output_csv, quoting=csv.QUOTE_ALL)

    def format(self, record):
        record_msg = record.getMessage()
        #print("Record message {}".format(record_msg))
        data = self.output.getvalue()
        self.output.truncate(0)
        self.output.seek(0)

        if 'global_step =' in record_msg :
            #Extract mAP values and steps
            #print("Global step section")
            mAP = record_msg.split(",")
            print("mAP {}".format(mAP))
            mAP_value = mAP[0].split(" = ")[1]
            mAP_step = mAP[-3].split(" = ")[1]
            print("mAP_value = {}, step= {}".format(mAP_value,mAP_step))
            #self.writer.writerow([str(mAP_step), str(mAP_value)])
        else:
            loss_step = record_msg.split("(")[0]
            #Strip the string of loss and steps
            print("loss_step {}".format(loss_step))
            loss,step = loss_step.split(",")[0].strip(),loss_step.split(",")[1].strip()
            loss_stripped = loss.split(" = ")[-1]
            step_stripped = step.split(" = ")[-1]
            print("step = {}, loss = {}".format(step_stripped,loss_stripped))
            #self.writer.writerow([str(step_stripped), str(loss_stripped)])
        return data.strip()


class StepLossFilter(logging.Filter):
    def filter(self, record):
        """
        Apply this logger’s filters to the record and return True if the record is to be processed. 
        The filters are consulted in turn, until one of them returns a false value. 
        If none of them return a false value, the record will be processed (passed to handlers). If one returns a false value, no further processing of the record occurs.
        """
        log_record = record.getMessage()
        #rint("log_records: {}".format(log_record))
        #Filter list
        filter_string = ["loss = ","step = ","Started","Ended","Duration","Loss for final step"]
        if any(terms in log_record for terms in filter_string):
          return True
        else:
          return False

#Set timezone to singapore
tz_SG = pytz.timezone("Asia/Singapore")
#Define tensorflow variable arguments

#Define arguments
flags.DEFINE_string('model_dir', None, 'Path to output model directory '
  'where event and checkpoint files will be written.')
flags.DEFINE_string('pipeline_config_path', None, 'Path to pipeline config '
                    'file.')
flags.DEFINE_integer('num_train_steps', None, 'Number of train steps.')
flags.DEFINE_boolean('eval_training_data', False,
                    'If training data should be evaluated for this job. Note '
                    'that one call only use this in eval-only mode, and '
                    '`checkpoint_dir` must be supplied.')
flags.DEFINE_integer('sample_1_of_n_eval_examples', 1, 'Will sample one of '
                    'every n eval input examples, where n is provided.')
flags.DEFINE_integer('sample_1_of_n_eval_on_train_examples', 5, 'Will sample '
                    'one of every n train input examples for evaluation, '
                    'where n is provided. This is only used if '
                    '`eval_training_data` is True.')
flags.DEFINE_string(
    'hparams_overrides', None, 'Hyperparameter overrides, '
    'represented as a string containing comma-separated '
    'hparam_name=value pairs.')
flags.DEFINE_string(
    'checkpoint_dir', None, 'Path to directory holding a checkpoint.  If '
    '`checkpoint_dir` is provided, this binary operates in eval-only mode, '
    'writing resulting metrics to `model_dir`.')
flags.DEFINE_boolean(
    'run_once', False, 'If running in eval-only mode, whether to run just '
    'one round of eval vs running continuously (default).'
)
FLAGS = flags.FLAGS

def main(unused_argv):
  flags.mark_flag_as_required('model_dir')
  flags.mark_flag_as_required('pipeline_config_path')
  current_time_start = datetime.now(tz_SG).strftime('%d-%m-%Y %H:%M:%S')
  start=time.time()
  
  log_directory = os.path.join(os.getcwd(),FLAGS.model_dir)

  #make directory for log files
  if not os.path.exists(log_directory):
    os.makedirs(log_directory)
  print("Logging will be found in {}".format(log_directory))
  log_file = os.path.join(log_directory, 'log.txt')

    # create file handler which logs event debug messages
  log = logging.getLogger('tensorflow')
  log.root.handlers[0].setFormatter(CsvFormatter(output_csv = os.path.join(log_directory, 'log.csv')))
  #log.disable(logging.WARNING)
  log.addFilter(StepLossFilter())
  config2 = tf.ConfigProto()
  config2.gpu_options.allow_growth = True
  log.setLevel(logging.INFO)
  #formatter = logging.Formatter('%(levelname)s - %(message)s')
  formatter = logging.Formatter('%(message)s')
  #FileHandler is used to send the log entries to a file
  fh = logging.FileHandler(log_file)
  print("File handler: {}".format(fh))
  fh.setLevel(logging.INFO)
  fh.setFormatter(formatter)
  log.addHandler(fh)

  # StreamHandler is used to send the log entries to console
  ch = logging.StreamHandler()
  ch.addFilter(StepLossFilter())
  ch.setLevel(logging.INFO)
  ch.setFormatter(formatter)
  log.addHandler(ch)
  
  #Log the estimator steps
  config = tf.estimator.RunConfig(model_dir=FLAGS.model_dir,save_checkpoints_steps=500, log_step_count_steps=100,session_config=config2)

  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))
  estimator = train_and_eval_dict['estimator']
  train_input_fn = train_and_eval_dict['train_input_fn']
  eval_input_fns = train_and_eval_dict['eval_input_fns']
  eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']
  predict_input_fn = train_and_eval_dict['predict_input_fn']
  train_steps = train_and_eval_dict['train_steps']

  if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
      estimator.evaluate(input_fn,
                         steps=None,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

    # Currently only a single Eval Spec is allowed.
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_specs[0])
  end=time.time()
  current_time_end = datetime.now(tz_SG).strftime('%d-%m-%Y %H:%M:%S')
  log.info("Started: {}".format(current_time_start))
  log.info("Ended: {}".format(current_time_end))
  log.info("Duration: {} secs".format(round(end-start,0)))

if __name__ == "__main__":
  tf.app.run()
