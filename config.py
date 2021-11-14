model_config = {
    'vae_mid': 20,
    'num_words': 29655,
    'vocab_size': 29655,
    'bow_mid_hid': 512,
    'seq_mid_hid': 512,
    'seq_len': 100,
    'num_heads': 8,
    'dropout': 0.8,
    'is_traing': True
}

train_config = {
    'batch_size': 1000,
    'epochs': 1000,
    'lr': 3e-3,
    'clip_grad': 20,
    'save_step': 100,
}

eval_config = {
    'batch_size': 1000,
}
