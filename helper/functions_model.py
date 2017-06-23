# Thanks to Mat Leonard (@MCLeopard) for the functions here
import time
from collections import namedtuple
import numpy as np
import tensorflow as tf

def split_data(chars, batch_size, num_steps, split_frac=0.9):
    """ 
    Split character data into training and validation sets, inputs and targets for each set.
    
    Arguments
    ---------
    chars: character array
    batch_size: Size of examples in each of batch
    num_steps: Number of sequence steps to keep in the input and pass to the network
    split_frac: Fraction of batches to keep in the training set
    
    Returns train_x, train_y, val_x, val_y
    """
    
    slice_size = batch_size * num_steps
    n_batches = int(len(chars) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = chars[: n_batches*slice_size]
    y = chars[1: n_batches*slice_size + 1]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the virst split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    
    return train_x, train_y, val_x, val_y

def get_batch(arrs, num_steps):
    batch_size, slice_size = arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]

def build_rnn(num_classes, batch_size=50, num_steps=50, lstm_size=128, num_layers=2,
              learning_rate=0.001, grad_clip=5, sampling=False):
        
    if sampling == True:
        batch_size, num_steps = 1, 1

    tf.reset_default_graph()
    
    # Declare placeholders we'll feed into the graph
    with tf.name_scope('inputs'):
        inputs = tf.placeholder(tf.int32, [batch_size, num_steps], name='inputs')
        x_one_hot = tf.one_hot(inputs, num_classes, name='x_one_hot')
    
    with tf.name_scope('targets'):
        targets = tf.placeholder(tf.int32, [batch_size, num_steps], name='targets')
        y_one_hot = tf.one_hot(targets, num_classes, name='y_one_hot')
        y_reshaped = tf.reshape(y_one_hot, [-1, num_classes])
    
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    # Build the RNN layers
    with tf.name_scope("RNN_cells"):
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    
    with tf.name_scope("RNN_init_state"):
        initial_state = cell.zero_state(batch_size, tf.float32)

    # Run the data through the RNN layers
    with tf.name_scope("RNN_forward"):
        outputs, state = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=initial_state)
    
    final_state = state
    
    # Reshape output so it's a bunch of rows, one row for each cell output
    with tf.name_scope('sequence_reshape'):
        seq_output = tf.concat(outputs, axis=1,name='seq_output')
        output = tf.reshape(seq_output, [-1, lstm_size], name='graph_output')
    
    # Now connect the RNN outputs to a softmax layer and calculate the cost
    with tf.name_scope('logits'):
        softmax_w = tf.Variable(tf.truncated_normal((lstm_size, num_classes), stddev=0.1),
                               name='softmax_w')
        softmax_b = tf.Variable(tf.zeros(num_classes), name='softmax_b')
        logits = tf.matmul(output, softmax_w) + softmax_b
        tf.summary.histogram('softmax_w', softmax_w)
        tf.summary.histogram('softmax_b', softmax_b)

    with tf.name_scope('predictions'):
        preds = tf.nn.softmax(logits, name='predictions')
        tf.summary.histogram('predictions', preds)
    
    with tf.name_scope('cost'):
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped, name='loss')
        cost = tf.reduce_mean(loss, name='cost')
        tf.summary.scalar('cost', cost)

    # Optimizer for training, using gradient clipping to control exploding gradients
    with tf.name_scope('train'):
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), grad_clip)
        train_op = tf.train.AdamOptimizer(learning_rate)
        optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    merged = tf.summary.merge_all()
    
    # Export the nodes 
    export_nodes = ['inputs', 'targets', 'initial_state', 'final_state',
                    'keep_prob', 'cost', 'preds', 'optimizer', 'merged']
    Graph = namedtuple('Graph', export_nodes)
    local_dict = locals()
    graph = Graph(*[local_dict[each] for each in export_nodes])
    
    return graph

# network helpers end

# lyric generator helpers
def pick_top_n(preds, vocab_size, top_n):
    p = np.squeeze(preds)
    p[np.argsort(p)[:-top_n]] = 0
    p = p / np.sum(p)
    c = np.random.choice(vocab_size, 1, p=p)[0]
    return c
    
def sample(checkpoint, n_samples, lstm_size, vocab_size, vocab_to_int, vocab, int_to_vocab, prime="The ", top_n=5):
    samples = [c for c in prime]
    model = build_rnn(vocab_size, lstm_size=lstm_size, sampling=True)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1, 1))
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

        c = pick_top_n(preds, len(vocab), top_n)
        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            preds, new_state = sess.run([model.preds, model.final_state], 
                                         feed_dict=feed)

            c = pick_top_n(preds, len(vocab), top_n)
            samples.append(int_to_vocab[c])
        
    return ''.join(samples)
    
# lyric generator helpers end