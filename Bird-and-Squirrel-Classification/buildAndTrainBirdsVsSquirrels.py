import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from preprocessDefinition import myPreprocess


def main():

    training_dataset = tf.data.TFRecordDataset(['./birds-vs-squirrels-train.tfrecords'])
    validation_dataset = tf.data.TFRecordDataset(['./birds-vs-squirrels-validation.tfrecords'])
    feature_description={'image':tf.io.FixedLenFeature([],tf.string),'label':tf.io.FixedLenFeature([],tf.int64)}

    def parse_examples(serialized_examples):
        examples=tf.io.parse_example(serialized_examples,feature_description)
        targets=examples.pop('label')
        images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'],channels=3),tf.float32),299,299)
        return images, targets

    train_dataset = training_dataset.map(parse_examples, num_parallel_calls=16).batch(64, drop_remainder=True)
    valid_dataset = validation_dataset.map(parse_examples, num_parallel_calls=16).batch(64, drop_remainder=True)

    normalizer = tf.keras.layers.Normalization(axis=-1)
    normalizer.adapt(train_dataset.map(lambda x, y: x))
    train_data = train_dataset.map(lambda x,y: normalizer(x))
    valid_data = valid_dataset.map(lambda x,y: normalizer(x))

    train_data = train_dataset.map(myPreprocess, num_parallel_calls=32).cache()
    valid_data = valid_dataset.map(myPreprocess, num_parallel_calls=32).cache()


    base_model=keras.applications.xception.Xception(weights='imagenet',include_top=False)
    #print(base_model.summary())

    avg = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    output = tf.keras.layers.Dense(3, activation="softmax")(avg)
    model = keras.Model(inputs=base_model.input, outputs=output)

    for layer in base_model.layers:
        layer.trainable = False


    # for layer in model.layers:
        # print(layer.trainable)


    earlyStop_cb=keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    lr = 5e-1
    optimizer = keras.optimizers.SGD(learning_rate=lr)


    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

    model.fit(train_data, validation_data=valid_data, epochs=20, callbacks=[earlyStop_cb])

    # model.save('modelmidway')
    #FITTING THE MODEL FOR LOWER LAYERS THIS TIME
    for layer in model.layers:
        model.trainable = True


    train_dataset = training_dataset.map(parse_examples, num_parallel_calls=16).batch(32, drop_remainder=True)
    valid_dataset = validation_dataset.map(parse_examples, num_parallel_calls=16).batch(32, drop_remainder=True)

    train_data = train_dataset.map(myPreprocess, num_parallel_calls=32).cache()
    valid_data = valid_dataset.map(myPreprocess, num_parallel_calls=32).cache()


    ss=3e-2
    optimizer=keras.optimizers.SGD(learning_rate=ss)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])


    earlyStop_cb=keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)
    model.fit(train_data, validation_data=valid_data, epochs=20, callbacks=[earlyStop_cb])

    model.save('birdsVsSquirrelsModel')


if __name__ == '__main__':
    main()

