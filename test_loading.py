import tensorflow as tf

def read_and_decode(filename_queue):
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)
    image_name = 'move/' + str(2) + '/image/encoded'
    ft = {image_name: tf.FixedLenFeature([1], tf.string)}

    features = tf.parse_single_example(
      serialized_example,
      features=ft)

    image_buffer = tf.reshape(features[image_name], shape=[])
    image = tf.image.decode_jpeg(image_buffer, channels=3)

    return image

tfrecord_filename = "./data/push_train.tfrecord-00257-of-00264"
filename_queue = tf.train.string_input_producer(
    [tfrecord_filename], num_epochs=1)
image = read_and_decode(filename_queue)
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session()  as sess:
    
    sess.run(init_op)
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(1):
    
        img = sess.run([image])
        print(type(img))
        print(img)

    coord.request_stop()
    coord.join(threads)
