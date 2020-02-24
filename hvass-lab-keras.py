import os
import argparse

import os
import subprocess as sp
import tqdm

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

import skimage.io as io
from PIL import Image
from pycocotools.coco import COCO


MARK_START = 'ssss '
MARK_END = 'eeee '


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download-data', action='store_true')
    parser.add_argument('--data-dir', type=str, default='/data')
    parser.add_argument('--num-words', type=int, default=10000)
    parser.add_argument('--state-size', type=int, default=512)
    parser.add_argument('--embedding-size', type=int, default=128)
    parser.add_argument('--image-batch-size', type=int, default=16)
    parser.add_argument('--text-batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--steps-per-epoch', type=int, default=1000)
    parser.add_argument('--train-set-size', type=int, default=None)
    parser.add_argument('--validation-set-size', type=int, default=None)
    parser.add_argument('--model-dir', type=str, default='/data')
    parser.add_argument('--model-name', type=str, default='model')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--load-model', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--show-images', action='store_true')
    return parser.parse_args()


def is_running_on_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    return len(physical_devices) != 0


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


def load_train_dataset(data_dir):
    train_data = COCO(f'{data_dir}/annotations/captions_train2017.json')
    return train_data


def load_validation_dataset(data_dir):
    val_data = COCO(f'{data_dir}/annotations/captions_val2017.json')
    return val_data


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


def show_images(image_ids, coco_data, image_dir, should_show_images):
    image_props_list = coco_data.loadImgs(image_ids)

    for image_props in image_props_list:
        if should_show_images:
            image = load_image(os.path.join(
                image_dir, image_props['file_name']))
            plt.axis('off')
            plt.imshow(image)
            plt.show()

        ann_ids = coco_data.getAnnIds(imgIds=image_props['id'])
        anns = coco_data.loadAnns(ann_ids)
        coco_data.showAnns(anns)


def create_image_model():
    image_model = VGG16(include_top=True, weights='imagenet')
    transfer_layer = image_model.get_layer('fc2')
    image_transer_model = Model(inputs=image_model.input,
                                outputs=transfer_layer.output)
    return image_transer_model


def create_batches(l, batch_size):
    for idx in range(0, len(l), batch_size):
        yield l[idx:idx+batch_size]


def get_images_ids(coco_data, set_size=None):
    image_ids_full = coco_data.getImgIds()
    return (
        image_ids_full if set_size is None
        else image_ids_full[:set_size]
    )


def process_images(image_ids, image_dir, coco_data,
                   model, batch_size=32, image_size=(224, 224),
                   transfer_size=4096):
    num_images = len(image_ids)
    image_shape = (batch_size,) + image_size + (3,)
    image_batch = np.zeros(shape=image_shape, dtype=np.float16)
    transfer_shape = (num_images, transfer_size)
    transfer_values = np.zeros(shape=transfer_shape, dtype=np.float16)
    image_id_batches = create_batches(image_ids, batch_size)
    start_index = 0
    for image_id_batch in tqdm(image_id_batches):
        image_props_batch = coco_data.loadImgs(ids=image_id_batch)
        # The size of the current batch can be less than batch_size
        # if the dataset doesn't devide evenly into batch_size
        current_batch_size = len(image_props_batch)
        for idx, image_props in enumerate(image_props_batch):
            image_batch[idx] = load_image(
                os.path.join(image_dir, image_props['file_name']),
                size=image_size)

        transfer_values_batch = model.predict(
            image_batch[0:current_batch_size])
        transfer_values[start_index:start_index + current_batch_size] = (
            transfer_values_batch[0:current_batch_size]
        )

        start_index += current_batch_size

    return transfer_values


def get_captions(coco_data, img_ids):
    anns_ids = coco_data.getAnnIds(imgIds=img_ids)
    anns = coco_data.loadAnns(ids=anns_ids)
    captions = []
    for img_id in img_ids:
        anns_ids = coco_data.getAnnIds(imgIds=img_id)
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


def create_decoder_model(transfer_values_size,
                         state_size,
                         embedding_size,
                         num_words):
    # Define the layers
    transfer_values_input = Input(shape=(transfer_values_size,),
                                  name='transfer_values_input')

    decoder_transfer_map = Dense(state_size,
                                 activation='tanh',
                                 name='decoder_transfer_map')

    decoder_input = Input(shape=(None, ), name='decoder_input')

    decoder_embedding = Embedding(input_dim=num_words,
                                  output_dim=embedding_size,
                                  name='decoder_embedding')

    decoder_gru1 = GRU(state_size, name='decoder_gru1',
                       return_sequences=True)
    decoder_gru2 = GRU(state_size, name='decoder_gru2',
                       return_sequences=True)
    decoder_gru3 = GRU(state_size, name='decoder_gru3',
                       return_sequences=True)

    # Dense is another term for Fully Connected
    decoder_dense = Dense(num_words,
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


def create_model(transfer_values_size,
                 state_size,
                 embedding_size,
                 learning_rate,
                 num_words,
                 load_model,
                 model_dir,
                 model_name):
    model = None
    # check if we can load model
    if load_model:
        pass
    else:
        # if not create, the model
        model = create_decoder_model(transfer_values_size,
                                    state_size,
                                    embedding_size,
                                    num_words)

        optimizer = RMSprop(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=sparse_cross_entropy)
    return model


def main():
    args = parse_args()
    image_dir = os.path.join(args.data_dir, 'images')
    if is_running_on_gpu():
        print('Using GPU')
    if args.download_data:
        download_data(args.data_dir, image_dir)

    coco_train = load_train_dataset(args.data_dir)

    tokenizer = create_tokenizer(coco_train, args.num_words)

    image_model = create_image_model()

    model = create_model(transfer_value_size,
                         args.state_size,
                         args.embedding_size,
                         args.num_words,
                         args.load_model,
                         args.model_dir,
                         args.model_name)
    if args.train:
        train(model, image_model, args.data_dir,
              args.save_model, args.model_name)


if __name__ == '__main__':
    main()
