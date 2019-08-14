from extract_training_data import FeatureExtractor
import sys
import numpy as np
import keras
from keras import Sequential
from keras.layers import Flatten, Embedding, Dense, Conv2D, MaxPooling2D, Dropout, Reshape

def build_model(word_types, pos_types, outputs, improve=False):
    # TODO: Write this function for part 3
    if improve == False:
        model = Sequential()
        model.add(Embedding(word_types, 32, input_length=6))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu'))
        model.add(Dense(units=10, activation='relu'))
        model.add(Dense(units=outputs, activation='softmax'))
        model.compile(keras.optimizers.Adam(lr=0.001), loss="categorical_crossentropy")
    else:
        model = Sequential()
        model.add(Embedding(word_types, 32, input_length=6))
        model.add(Reshape((1, 6, 32)))
        # embed 15153 word types to 32 slots, each represented by a length 6 array
        model.add(Conv2D(filters=8, kernel_size = (3,3), border_mode='same', activation='relu', input_shape=(6,32,1)))
        model.add(MaxPooling2D(1,1))
        model.add(Flatten())
        model.add(Dense(units=100, activation='relu'))
        model.add(Dropout(rate=0.2))
        model.add(Dense(units=50, activation='relu'))
        model.add(Dense(units=outputs, activation='softmax'))
        model.compile(keras.optimizers.Adam(lr=0.001), loss="categorical_crossentropy")
    return model


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r') 
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1) 

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    print("Compiling model.")
    model = build_model(len(extractor.word_vocab), len(extractor.pos_vocab), len(extractor.output_labels))
    inputs = np.load(sys.argv[1])
    outputs = np.load(sys.argv[2])
    print("Done loading data.")
   
    # Now train the model
    model.fit(inputs, outputs, epochs=5, batch_size=100)
    
    model.save(sys.argv[3])
