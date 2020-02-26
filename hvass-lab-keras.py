import os
import argparse

import os
import subprocess as sp
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import sentence_bleu

import skimage.io as io
from PIL import Image
from pycocotools.coco import COCO


MARK_START = 'ssss '
MARK_END = 'eeee '


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--data-dir', type=str, default='/data')
    parser.add_argument('--model-dir', type=str, default='/data')
    parser.add_argument('--model-name', type=str, default='model')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train-data-size', type=int, default=None)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--validate-data-size', type=int, default=None)
    parser.add_argument('--state-size', type=int, default=512)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--num-words', type=int, default=10000)
    parser.add_argument('--image-batch-size', type=int, default=32)
    parser.add_argument('--text-batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--show-captions', action='store_true')
    parser.add_argument('--max-words', type=int, default=30)
    return parser.parse_args()


def is_running_on_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    return len(gpus) != 0


def configure_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    # For now we are assuming we are only using 1 GPU
    tf.config.experimental.set_memory_growth(gpus[0], False)
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])


def get_image_dir(data_dir):
    return os.path.join(data_dir, 'images')


def get_captions(coco_data, image_ids):
    anns_ids = coco_data.getAnnIds(imgIds=image_ids)
    anns = coco_data.loadAnns(ids=anns_ids)
    captions = []
    for image_id in image_ids:
        anns_ids = coco_data.getAnnIds(imgIds=image_id)
        anns = coco_data.loadAnns(ids=anns_ids)
        captions.append([ann['caption'] for ann in anns])
    return captions


def mark_captions(captions):
    captions_marked = [[MARK_START + caption + MARK_END
                        for caption in captions_for_one_img]
                       for captions_for_one_img in captions]
    return captions_marked


def flatten(captions):
    captions_flat = [caption
                     for captions_for_one_img in captions
                     for caption in captions_for_one_img]
    return captions_flat


def download_and_extract(file_name_no_extension, dest_dir=None, data_dir='.', sub_path='zips'):
    '''Download and extract zip files if they haven't been already.
    '''
    print(f'Downloading {file_name_no_extension}.zip')
    sp.call(
        f'wget --timestamping -c -P {data_dir} http://images.cocodataset.org/{sub_path}/{file_name_no_extension}.zip', shell=True)
    if dest_dir:
        os.makedirs(data_dir, exist_ok=True)
        print(f'Extracting {file_name_no_extension}.zip to {dest_dir}')
        sp.call(
            f'unzip -uojq {data_dir}/{file_name_no_extension}.zip -d {dest_dir}', shell=True)
    else:
        print(f'Extracting {file_name_no_extension}.zip to {data_dir}')
        sp.call(
            f'unzip -uoq {data_dir}/{file_name_no_extension}.zip -d {data_dir}', shell=True)


def download_data(data_dir, image_dir):
    print('Downloading data')
    download_and_extract('annotations_trainval2017',
                         data_dir=data_dir, sub_path='annotations')
    download_and_extract('train2017', dest_dir=image_dir, data_dir=data_dir)
    download_and_extract('val2017', dest_dir=image_dir, data_dir=data_dir)
    print('Finished downloading data')


def load_image(path, size=None):
    """
    Load the image from the given file-path and resize it
    to the given size if not None.
    """
    image = Image.open(path)
    # Resize image if desired.
    if not size is None:
        image = image.resize(size=size, resample=Image.LANCZOS)
    image = np.array(image)
    # Scale image-pixels so they fall between 0.0 and 1.0
    image = image / 255.0
    # Convert 2-dim gray-scale array to 3-dim RGB array.
    if (len(image.shape) == 2):
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    return image


def show_images(image_ids, coco_data, image_dir):
    image_props_list = coco_data.loadImgs(image_ids)

    for image_props in image_props_list:
        image = load_image(os.path.join(
            image_dir, image_props['file_name']))
        plt.axis('off')
        plt.imshow(image)
        plt.show()


def show_captions(image_ids, coco_data):
    ann_ids = coco_data.getAnnIds(imgIds=image_ids)
    anns = coco_data.loadAnns(ann_ids)
    coco_data.showAnns(anns)


def create_batches(l, batch_size):
    for idx in range(0, len(l), batch_size):
        yield l[idx:idx+batch_size]


def batch_generator(num_images, transfer_values, tokens, batch_size):
    """
    Generator function for creating random batches of training-data.

    Note that it selects the data completely randomly for each
    batch, corresponding to sampling of the training-set with
    replacement. This means it is possible to sample the same
    data multiple times within a single epoch - and it is also
    possible that some data is not sampled at all within an epoch.
    However, all the data should be unique within a single batch.
    """

    while True:
        random_image_indices = np.random.randint(num_images, size=batch_size)
        tv = transfer_values[random_image_indices]
        random_tokens = get_random_caption_tokens(random_image_indices, tokens)
        num_tokens = [len(t) for t in random_tokens]
        max_tokens = np.max(num_tokens)
        tokens_padded = pad_sequences(random_tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')

        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]

        x_data = {
            'decoder_input': decoder_input_data,
            'transfer_values_input': tv
        }

        y_data = {
            'decoder_output': decoder_output_data
        }

        yield (x_data, y_data)


def get_images_ids(coco_data, set_size=None):
    image_ids = coco_data.getImgIds()
    if set_size is None:
        return image_ids
    else:
        return image_ids[:set_size]


def create_image_model():
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_transer_model = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    return image_transer_model


def get_transfer_value_size(image_model):
    return K.int_shape(image_model.output)[1]


def get_image_size(image_model):
    return K.int_shape(image_model.input)[1:3]


def get_bleu_score(captions, predicted_caption):
    captions_as_lists = []
    for caption in captions:
        caption_as_list = caption.split(' ')
        captions_as_lists.append(caption_as_list)
    predicted_caption_list = predicted_caption.split(' ')
    return sentence_bleu(captions_as_lists, predicted_caption_list)


def sparse_cross_entropy(y_true, y_pred):
    """
    Calculate the cross-entropy loss between y_true and y_pred.

    y_true is a 2-rank tensor with the desired output.
    The shape is [batch_size, sequence_length] and it
    contains sequences of integer-tokens.

    y_pred is the decoder's output which is a 3-rank tensor
    with shape [batch_size, sequence_length, num_words]
    so that for each sequence in the batch there is a one-hot
    encoded array of length num_words.
    """

    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True)
    # Keras may reduce this across the first axis (the batch)
    # but the semantics are unclear, so to be sure we use
    # the loss across the entire 2-rank tensor, we reduce it
    # to a single scalar with the mean function.
    loss_mean = tf.reduce_mean(loss)
    return loss_mean


def get_random_caption_tokens(image_indices, tokens):
    """
    Given a list of indices for images in the training-set,
    select a token-sequence for a random caption for each image,
    and return a list of all these token-sequences.
    """
    result = []
    for image_index in image_indices:
        caption_index = np.random.choice(len(tokens[image_index]))
        random_tokens = tokens[image_index][caption_index]
        result.append(random_tokens)
    return result


class TokenizerWrap(Tokenizer):
    """Wrap the Tokenizer-class from Keras with more functionality."""

    def __init__(self, texts, num_words=None):
        """
        :param texts: List of strings with the data-set.
        :param num_words: Max number of words to use.
        """
        super().__init__(num_words=num_words)

        # Create the vocabulary from the texts.
        self.fit_on_texts(texts)

        # Create inverse lookup from integer-tokens to words.
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        """Lookup a single word from an integer-token."""

        word = " " if token == 0 else self.index_to_word[token]
        return word

    def tokens_to_string(self, tokens):
        """Convert a list of integer-tokens to a string."""

        # Create a list of the individual words.
        words = [self.index_to_word[token]
                 for token in tokens
                 if token != 0]

        # Concatenate the words to a single string
        # with space between all the words.
        text = " ".join(words)

        return text

    def captions_to_tokens(self, captions_listlist):
        """
        Convert a list-of-list with text-captions to
        a list-of-list of integer-tokens.
        """

        # Note that text_to_sequences() takes a list of texts.
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]

        return tokens


def create_tokenizer(coco_data, num_words):
    image_ids = get_images_ids(coco_data)
    captions = get_captions(coco_data, image_ids)
    marked_captions = mark_captions(captions)
    texts = flatten(marked_captions)
    return TokenizerWrap(texts, num_words)


class CaptionItModelRunner(object):
    def __init__(self,
                 data_dir, model_dir, model_name,
                 save_model, load_model,
                 train_data_size, val_data_size,
                 state_size, embedding_size, num_words,
                 image_batch_size, text_batch_size, learning_rate):
        super().__init__()
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.model_name = model_name
        self.save_model = save_model
        self.load_model = load_model
        self.train_data_size = train_data_size
        self.val_data_size = val_data_size
        self.state_size = state_size
        self.embedding_size = embedding_size
        self.num_words = num_words
        self.image_batch_size = image_batch_size
        self.text_batch_size = text_batch_size
        self.learning_rate = learning_rate

        self.image_dir = get_image_dir(data_dir)

        # Populated in initialize
        self.coco_data_train = None
        self.coco_data_val = None
        self.train_image_ids = None
        self.val_image_ids = None
        self.image_model = None
        self.decoder_model = None
        self.tokenizer = None
        self.transfer_value_size = None
        self.image_size = None

    def initialize(self):
        self.coco_data_train = self._load_train_dataset()
        self.coco_data_val = self._load_validate_dataset()
        self.train_image_ids = get_images_ids(
            self.coco_data_train, self.train_data_size)
        self.train_data_size = len(self.train_image_ids)
        self.val_image_ids = get_images_ids(
            self.coco_data_val, self.val_data_size)
        self.val_data_size = len(self.val_image_ids)
        self.image_model = create_image_model()
        self.transfer_value_size = get_transfer_value_size(self.image_model)
        self.image_size = get_image_size(self.image_model)
        self.decoder_model = self._create_decoder_model()
        self.tokenizer = create_tokenizer(self.coco_data_train,
                                          self.num_words)

    def train(self, epochs, steps_per_epoch):
        transfer_values = self._process_images(self.train_image_ids,
                                               self.coco_data_train)
        captions = get_captions(self.coco_data_train,
                                self.train_image_ids)
        marked_captions = mark_captions(captions)
        tokens = self.tokenizer.captions_to_tokens(marked_captions)
        num_images = len(self.train_image_ids)
        generator = batch_generator(num_images=num_images,
                                    transfer_values=transfer_values,
                                    tokens=tokens,
                                    batch_size=self.text_batch_size)
        self.decoder_model.fit(generator,
                               epochs=epochs,
                               steps_per_epoch=steps_per_epoch)

    def generate_caption(self, image_path_or_id, max_words=30, coco_data=None):
        image = None
        image_path = None
        if os.path.exists(image_path_or_id):
            image_path = image_path_or_id
        else:
            image_props = coco_data.loadImgs(image_path_or_id)
            image_path = os.path.join(self.image_dir,
                                      image_props[0]['file_name'])

        image = load_image(image_path, self.image_size)
        image_as_batch = np.expand_dims(image, axis=0)
        transfer_values = self.image_model.predict(image_as_batch)
        decoder_input_data = np.zeros(shape=(1, max_words),
                                      dtype=np.int)
        token_start = self.tokenizer.word_index[MARK_START.strip()]
        token_end = self.tokenizer.word_index[MARK_END.strip()]
        token_int = token_start
        output_text = ''
        count_tokens = 0

        while token_int != token_end and count_tokens < max_words:
            decoder_input_data[0, count_tokens] = token_int
            x_data = {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }

            decoder_output = self.decoder_model.predict(x_data)
            token_onehot = decoder_output[0, count_tokens, :]
            token_int = np.argmax(token_onehot)
            sampled_word = self.tokenizer.token_to_word(token_int)
            output_text += ' ' + sampled_word
            count_tokens += 1

        output_text = output_text.replace(MARK_END, '').strip()
        output_text = output_text.capitalize() + '.'
        return output_text

    def validate(self, should_show_captions, max_words=30):
        bleu_scores = []
        for image_id in self.val_image_ids:
            captions = get_captions(self.coco_data_val, [image_id])[0]
            predicted_caption = self.generate_caption(
                image_id, max_words, self.coco_data_val)
            bleu_score = get_bleu_score(captions, predicted_caption)
            bleu_scores.append(bleu_score)
            if should_show_captions:
                print('Ground truth captions:')
                show_captions(image_id, self.coco_data_val)
                print('Predicted caption:')
                print(predicted_caption)
                print('Bleu Score:')
                print(bleu_score)
                print()

        print(f'Avg bleu score:     {np.mean(bleu_scores)}')
        print(f'Median bleu score:  {np.median(bleu_scores)}')

    def _load_train_dataset(self):
        train_data = COCO(
            f'{self.data_dir}/annotations/captions_train2017.json')
        return train_data

    def _load_validate_dataset(self):
        val_data = COCO(f'{self.data_dir}/annotations/captions_val2017.json')
        return val_data

    def _process_images(self, image_ids, coco_data):
        num_images = len(image_ids)
        image_shape = (self.image_batch_size,) + self.image_size + (3,)
        image_batch = np.zeros(shape=image_shape, dtype=np.float16)
        transfer_shape = (num_images, self.transfer_value_size)
        transfer_values = np.zeros(shape=transfer_shape, dtype=np.float16)
        image_id_batches = create_batches(image_ids, self.image_batch_size)
        start_index = 0
        for image_id_batch in tqdm(image_id_batches):
            image_props_batch = coco_data.loadImgs(ids=image_id_batch)
            # The size of the current batch can be less than batch_size
            # if the dataset doesn't divide evenly into batch_size
            current_batch_size = len(image_props_batch)
            for idx, image_props in enumerate(image_props_batch):
                image_path = os.path.join(
                    self.image_dir, image_props['file_name'])
                image_batch[idx] = load_image(image_path, size=self.image_size)

            transfer_values_batch = self.image_model.predict(
                image_batch[0:current_batch_size])
            transfer_values[start_index:start_index + current_batch_size] = (
                transfer_values_batch[0:current_batch_size]
            )

            start_index += current_batch_size

        return transfer_values

    def _create_decoder_network(self):
        # Define the layers
        transfer_values_input = Input(shape=(self.transfer_value_size,),
                                      name='transfer_values_input')

        decoder_transfer_map = Dense(self.state_size,
                                     activation='tanh',
                                     name='decoder_transfer_map')

        decoder_input = Input(shape=(None, ), name='decoder_input')

        decoder_embedding = Embedding(input_dim=self.num_words,
                                      output_dim=self.embedding_size,
                                      name='decoder_embedding')

        decoder_gru1 = GRU(self.state_size, name='decoder_gru1',
                           return_sequences=True)
        decoder_gru2 = GRU(self.state_size, name='decoder_gru2',
                           return_sequences=True)
        decoder_gru3 = GRU(self.state_size, name='decoder_gru3',
                           return_sequences=True)

        # Dense is another term for Fully Connected
        decoder_dense = Dense(self.num_words,
                              activation='linear',
                              name='decoder_output')

        # Map the transfer-values so the dimensionality matches
        # the internal state of the GRU layers. This means
        # we can use the mapped transfer-values as the initial state
        # of the GRU layers.
        initial_state = decoder_transfer_map(transfer_values_input)

        # Start the decoder-network with its input-layer.
        net = decoder_input

        # Connect the embedding-layer.
        net = decoder_embedding(net)

        # Connect all the GRU layers.
        net = decoder_gru1(net, initial_state=initial_state)
        net = decoder_gru2(net, initial_state=initial_state)
        net = decoder_gru3(net, initial_state=initial_state)

        # Connect the final dense layer that converts to
        # one-hot encoded arrays.
        decoder_output = decoder_dense(net)

        decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                              outputs=[decoder_output])

        return decoder_model

    def _create_decoder_model(self):
        model = None
        # check if we can load model
        if self.load_model:
            pass
        else:
            model = self._create_decoder_network()
            optimizer = RMSprop(learning_rate=self.learning_rate)
            model.compile(optimizer=optimizer, loss=sparse_cross_entropy)
        return model


def main():
    args = parse_args()
    image_dir = os.path.join(args.data_dir, 'images')

    if is_running_on_gpu():
        print('Using GPU')
        configure_gpu()
    if args.download_data:
        download_data(args.data_dir, image_dir)

    model_runner = CaptionItModelRunner(data_dir=args.data_dir,
                                        model_dir=args.model_dir,
                                        model_name=args.model_name,
                                        save_model=args.save_model,
                                        load_model=args.load_model,
                                        train_data_size=args.train_data_size,
                                        val_data_size=args.validate_data_size,
                                        state_size=args.state_size,
                                        embedding_size=args.embedding_size,
                                        num_words=args.num_words,
                                        image_batch_size=args.image_batch_size,
                                        text_batch_size=args.text_batch_size,
                                        learning_rate=args.learning_rate)

    model_runner.initialize()

    if args.train:
        model_runner.train(args.epochs,
                           args.steps_per_epoch)

    if args.validate:
        model_runner.validate(args.show_captions, args.max_words)


if __name__ == '__main__':
    main()
