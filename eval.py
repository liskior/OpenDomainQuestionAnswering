import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np
import tensorflow_datasets as tfds
from transformers import BertTokenizer
from transformers.data.processors.squad import SquadResult, SquadV1Processor
from absl import app
import tensorflow_hub as hub
from tensorflow.keras.layers import Layer
import os
import sys
from tqdm import tqdm
from collections import Counter

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class BertLayer(tf.keras.layers.Layer):
   def __init__(
      self,
      bert_path="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
      **kwargs
   ):
      self.n_fine_tune_layers = 3
      self.trainable  = True
      #self.output_size = 768
      self.bert_path = bert_path

      super(BertLayer, self).__init__(**kwargs)

   def build(self, input_shape):
      self.bert = hub.load(
         self.bert_path
      )

      # Remove unused layers
      trainable_vars = self.bert.variables
      trainable_layers = []
      non_trainable_layers = []

      # Select how many layers to fine tune
      for i in range(self.n_fine_tune_layers):
         trainable_layers.append("encoder/layer_{}".format(11 - i))

      # Update trainable vars to contain only the specified layers
      non_trainable_vars = [
         var
         for var in trainable_vars
            if all([l not in var.name for l in trainable_layers])
      ]

      trainable_vars = [
         var
         for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
      ]

      for var in trainable_vars:
         self._trainable_weights.append(var)


      for var in non_trainable_vars:
            self._non_trainable_weights.append(var)

      super(BertLayer, self).build(input_shape)

   def call(self, inputs):
      inputs = [K.cast(x, dtype="int32") for x in inputs]
      input_ids, input_mask, segment_ids = inputs
      bert_inputs = [
         input_ids, input_mask, segment_ids
      ]
      out = self.bert(inputs=bert_inputs)[1]
      return out

def build_model(max_seq_length, learning_rate):
   vocab_size = len(tokenizer.get_vocab())

   input_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_word_ids")
   input_mask = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="input_mask")
   segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), dtype=tf.int32, name="segment_ids")

   emb = tf.keras.layers.Embedding(input_dim=128, output_dim=128, mask_zero=True)
   bert_inputs = [input_ids, input_mask, segment_ids]
   bert_output = BertLayer()(bert_inputs)

   dense = tf.keras.layers.Dense(vocab_size, activation='softmax')
   td = tf.keras.layers.TimeDistributed(dense, input_shape=(max_seq_length, 768))(bert_output, mask=emb.compute_mask(input_mask))

   model = tf.keras.models.Model(inputs=bert_inputs, outputs=td)
   model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), metrics=['accuracy'])
   model.summary()

   return model

model = build_model(128, 0.00005)
#tf.keras.utils.plot_model(model, to_file="model.png", show_shapes=True)

def load_data():
    data, info = tfds.load("squad", with_info=True)
    print(info)
    data = SquadV1Processor().get_examples_from_dataset(data, evaluate=False)
    return data


def tokenize(data, function):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    data = map(function, data)
    data = list(data)
    data = map(tokenizer.tokenize, data)
    data = list(data)
    result = []
    for item in data:
        tokens = []
        tokens.append("[CLS]")
        tokens.extend(item)
        tokens.append("[SEP]")
        result.append(tokens)

    return result


def tokenize_test(data, function):

    data = map(function, data)
    data = list(data)
    data = map(tokenizer.tokenize, data)
    data = list(data)
    result = []
    for item in data:
        tokens = []
        tokens.extend(item)
        result.append(tokens)

    return result

def remove_non_digit(answers, questions):
    res_ans = []
    res_qwe = []
    for i in range(len(answers)):
        if (len(answers[i]) != 3): continue
        for j in answers[i]:
            if j.isnumeric():
                res_ans.append(answers[i])
                if questions is not None:
                    res_qwe.append(questions[i])
                break
    return res_ans, res_qwe


def prep_test(data, max_seq_length):

    get_questions = lambda q: q.question_text
    get_answers = lambda q: q.answer_text

    answers = tokenize(data, get_answers)
    questions = tokenize(data, get_questions)
    print(1, len(answers), len(questions))
    answers, questions = remove_non_digit(answers, questions)
    print(2, len(answers), len(questions))
    print(answers)
    masks = []
    segments = []

    for i in range(len(answers)):
       mask = []
       segment = []

       segment.extend([0] * len(questions[i]))
       segment.extend([1] * (len(answers[i]) - 1))
       segment.extend([0] * (max_seq_length - len(segment)))
       segments.append(segment)

       questions[i].extend(["[MASK]"] * (len(answers[i]) - 2) + ["[SEP]"])
       mask.extend([1] * len(questions[i]))
       mask.extend([0] * (max_seq_length - len(mask)))
       masks.append(mask)


    questions = map(lambda q: tokenizer.convert_tokens_to_ids(q), questions)
    questions = list(questions)
    answers = tokenize_test(data, get_answers)
    answers, _ = remove_non_digit(answers, None)
    answers = map(lambda q: tokenizer.convert_tokens_to_ids(q), answers)

    answers = list(answers)
    print(3, len(answers), len(questions))

    for i in range(len(answers)):
       questions[i].extend([0] * (max_seq_length - len(questions[i])))

    return questions, masks, segments, answers

data = load_data()

test_data = data[int(len(data) * 0.8):]

test_input_ids, test_masks, test_segments, test_answers = prep_test(test_data, 128)

print(len(test_answers))


def prepare_to_eval(x, y, z, ans):
   res = []
   for i in range(len(x)):
      tokens = np.asarray([x[i]])
      masks = np.asarray([y[i]])
      segments = np.asarray([z[i]])
      res.append([tokens, masks, segments])
   answers = np.asarray(ans)
   return res, answers

test_inputs, test_answers = prepare_to_eval(test_input_ids, test_masks, test_segments, test_answers)


checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)
tb_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")

class F1_metrics(tf.keras.callbacks.Callback):
   def on_epoch_end(self, epoch, logs=None):
      myeval()
      #print('The average loss for epoch {} is {:7.2f} and mean absolute error is {:7.2f}.'.format(epoch, logs['loss'], logs['mae']))

evaluate = F1_metrics()

def model_out(x):
   seg = x[2]
   print(x)
   y = model.predict(x)
   y = np.argmax(y, axis=-1)
   y = np.multiply(y[0], seg)

   y = np.trim_zeros(np.asarray(y[0]))
   #print(y)
   return y


def normalize_answer(s):
    def remove_articles(s):
        art = "a an the of"
        art = tokenizer.tokenize(art)
        art = tokenizer.convert_tokens_to_ids(art)
        for a in art:
           s = s[s != a]
        return s

    def remove_punc(s):
        art = ". , : - ' \" / ! ?"
        art = tokenizer.tokenize(art)
        art = tokenizer.convert_tokens_to_ids(art)
        for a in art:
           s = s[s != a]
        return s
    return remove_articles(remove_punc(s))

def f1_score(prediction, ground_truth):
    prediction = np.resize(prediction, prediction.size - 1)
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    print(tokenizer.convert_ids_to_tokens(prediction), tokenizer.convert_ids_to_tokens(ground_truth))
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def myeval():
   f1 = 0
   with tqdm(total=len(test_inputs), desc="In progress", bar_format="{l_bar}{bar} [ time left: {remaining} ]") as pbar:
      for i in range(len(test_inputs)):
         f1 += f1_score(model_out(test_inputs[i]), np.asarray(test_answers[i]))
         #print(f1)
         pbar.update(1)
   f1 = 100.0 * f1 / len(test_inputs)
   print("F1 score: ", f1, )
print(len(test_inputs))

checkpoint_path = "training/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.load_weights(checkpoint_path)

myeval()
