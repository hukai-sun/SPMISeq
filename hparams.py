import tensorflow as tf 

hparams = tf.contrib.training.HParams(
    # Trans network
    word_dim = 100,
    char_dim = 30,
    num_filters = 30,
    kernel_size = 3,
    Trans_hidden_size = 200,
    Trans_rnn_layers = 1,
    
    # Pred network
    label_dim = 10,
    label_embedd_type = 'one_hot', 
    Pred_hidden_size = 50,
    Pred_rnn_layers = 1,

    jt_hidden_size = 100,
    sos = 0,    
    dropout_ratio = 0.5,
    unk_replace = 0.0,   

    optimizer = 'SGD',    
    learning_rate = 1e-2,
    decay_rate = 0.05,    
    schedule = 1, 
    clip_grad = False,
    clip_grad_norm = 5.0,

    batch_size = 16,
    num_epochs = 200,
    early_start = 5,
    early_patience = 10,

    beam_width = 128,
    num_samples = 128
)

def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)