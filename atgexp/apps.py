from django.apps import AppConfig
import os
from django.conf import settings
import cv2
import json
from sklearn.utils import shuffle
import numpy as np
import tensorflow as tf


from .decoder_class import RNN_Decoder

class AtgexpConfig(AppConfig):
    name = 'atgexp'
    
    
    ### yolo configurations #####
    yolo = "yolo-coco"
    
    labelsPath = os.path.join(settings.MODELS, 'coco.names')
    weightsPath = os.path.join(settings.MODELS,'yolov3.weights')
    configPath = os.path.join(settings.MODELS,'yolov3.cfg') 
    
    LABELS = open(labelsPath).read().strip().split("\n")
    
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ### Yolo configurations complete ####
    
    
    ### GRU configurations ###
    annotation_file = os.path.join(settings.MODELS,'caption_gen')
    
    annotation_file = os.path.join(annotation_file,'captions_train2014.json')
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)
    
    all_captions = []
    all_img_name_vector = []    
    PATH = os.path.join(settings.MEDIA_ROOT,'images')
    
    
    for annot in annotations['annotations']:
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)
    
        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)
    
    train_captions, img_name_vector = shuffle(all_captions,
                                          all_img_name_vector,
                                          random_state=1)

    num_examples = 50000
    train_captions = train_captions[:num_examples]   
    print(len(train_captions))
    
    max_length = 49
    attention_features_shape = 64
    
    @staticmethod
    def load_image(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img,channels=3)
        img = tf.image.resize(img,(299,299))
        img = tf.keras.applications.inception_v3.preprocess_input(img)
        return img, image_path
    
    
    image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    
    image_features_extract_model = tf.keras.Model(new_input,hidden_layer)
    
    # choose top 5000 words from vocab
    tok_k = 5000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words = tok_k,
                                                      oov_token = '<unk>',
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ '
                                                      )
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    
    
    
    #
    embedding_dim = 256
    units = 512
    top_k = 5000
    vocab_size = top_k + 1
    
    gru_path = os.path.join(settings.MODELS,'caption_gen','saved2')
    
    encoder = tf.keras.models.load_model(os.path.join(gru_path,'encoder.hd5'),
                                         custom_objects=None,
                                         compile=True
                                        )
    
    decoder = RNN_Decoder(embedding_dim, units, vocab_size)
    decoder.load_weights(os.path.join(gru_path,'decoder.hd5'))
    
    @staticmethod
    def evaluate(image):
        attention_plot = np.zeros((AtgexpConfig.max_length, AtgexpConfig.attention_features_shape))
        hidden = AtgexpConfig.decoder.reset_state(batch_size=1)
        
        temp_input = tf.expand_dims(AtgexpConfig.load_image(image)[0], 0)
        img_tensor_val = AtgexpConfig.image_features_extract_model(temp_input)
        img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
        
        features = AtgexpConfig.encoder(img_tensor_val)
        dec_input = tf.expand_dims([AtgexpConfig.tokenizer.word_index['<start>']], 0)
        result = []
        
        for i in range(AtgexpConfig.max_length):
            predictions, hidden, attention_weights = AtgexpConfig.decoder(dec_input, features, hidden)
            attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
            predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
            result.append(AtgexpConfig.tokenizer.index_word[predicted_id])
            
            if AtgexpConfig.tokenizer.index_word[predicted_id] == '<end>':
                return result, attention_plot
            
            dec_input = tf.expand_dims([predicted_id], 0)
        
        attention_plot = attention_plot[:len(result), :]
        return result, attention_plot
        
    ### GRU configurations complete ###