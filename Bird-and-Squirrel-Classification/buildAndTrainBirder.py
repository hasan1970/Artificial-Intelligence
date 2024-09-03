import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
from preprocessDefinition import myPreprocess


def main():
    training_dataset = tf.data.TFRecordDataset(['./birds-20-eachOf-358.tfrecords'])
    validation_dataset = tf.data.TFRecordDataset(['./birds-10-eachOf-358.tfrecords'])
    feature_description={'image':tf.io.FixedLenFeature([],tf.string),'birdType':tf.io.FixedLenFeature([],tf.int64)}


    def parse_examples(serialized_examples):
        examples=tf.io.parse_example(serialized_examples,feature_description)
        targets=examples.pop('birdType')
        images=tf.image.resize_with_pad(tf.cast(tf.io.decode_jpeg(examples['image'],channels=3),tf.float32),299,299)
        return images, targets


    train_dataset = training_dataset.map(parse_examples, num_parallel_calls=16).batch(128, drop_remainder=True)
    valid_dataset = validation_dataset.map(parse_examples, num_parallel_calls=16).batch(128, drop_remainder=True)

    with open('birdNames.txt', 'r') as file:
        bird_names = [line.strip() for line in file.readlines()]
    label_to_name = {index: name for index, name in enumerate(bird_names)}

    train_data = train_dataset.map(myPreprocess, num_parallel_calls=16).cache()
    valid_data = valid_dataset.map(myPreprocess, num_parallel_calls=16).cache()

    rand_mods=tf.keras.models.Sequential([
        tf.keras.layers.Input((299,299,3)),
        tf.keras.layers.RandomContrast([.5,1.25]),
        tf.keras.layers.RandomZoom(.2),
        tf.keras.layers.RandomTranslation(height_factor=.2,width_factor=.2,fill_mode='constant'),
        tf.keras.layers.RandomRotation(.2)
    ])


    res=tf.keras.layers.Lambda(tf.keras.applications.xception.preprocess_input)(rand_mods.output)


    base_model=tf.keras.applications.xception.Xception(weights='imagenet',include_top=False)
    #print(base_model.summary())
    xout = base_model(res)
    avg = tf.keras.layers.GlobalAveragePooling2D()(xout)
    output = tf.keras.layers.Dense(358, activation="softmax")(avg)
    model = tf.keras.Model(inputs=rand_mods.input, outputs=output)


    for layer in base_model.layers:
        layer.trainable = False


    earlyStop_cb=keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    lr = 5e-1
    optimizer = keras.optimizers.SGD(learning_rate=lr)


    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])


    model.fit(train_data, validation_data=valid_data, epochs=25, callbacks=[earlyStop_cb])

    for layer in base_model.layers:
        layer.trainable = True

    # for layer in model.layers:
    #     print(layer.trainable)


    train_dataset = training_dataset.map(parse_examples, num_parallel_calls=16).batch(32, drop_remainder=True)
    valid_dataset = validation_dataset.map(parse_examples, num_parallel_calls=16).batch(32, drop_remainder=True)


    train_data = train_dataset.map(myPreprocess, num_parallel_calls=32).cache()
    valid_data = valid_dataset.map(myPreprocess, num_parallel_calls=32).cache()


    ss=3e-2
    optimizer=keras.optimizers.SGD(learning_rate=ss)
    model.compile(loss="sparse_categorical_crossentropy",optimizer=optimizer,metrics=["accuracy"])

    earlyStop_cb=keras.callbacks.EarlyStopping(patience=6,restore_best_weights=True)
    model.fit(train_data, validation_data=valid_data, epochs=30, callbacks=[earlyStop_cb])

    model.save('birder')

if __name__ == '__main__':
    main()





