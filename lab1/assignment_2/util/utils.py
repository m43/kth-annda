import tensorflow as tf


def generate_in_out_tensors(inputs, outputs, start, end, verbose=False):
    in_tensor = inputs[start:end]
    out_tensor = outputs[start:end]

    in_tensor = tf.convert_to_tensor(in_tensor, dtype='float64')
    out_tensor = tf.convert_to_tensor(out_tensor, dtype='float64')

    if verbose:
        print('Generated inputs:', len(in_tensor))
        print(in_tensor)
        print('Generated outputs:', len(out_tensor))
        print(out_tensor)

    return in_tensor, out_tensor
