global_params = {
    'down_width': 360,
    'down_height': 360,
    'embedding_size': 128,
    'n_classes': 7,

    'class_targets': {
        'bags': 0,
        'full_body': 1, 
        'glasses': 2,
        'lower_body': 3,
        'other': 4, 
        'shoes': 5,
        'upper_body': 6
    },

    'target_classes': {
        0: 'bags',
        1: 'full_body', 
        2: 'glasses',
        3: 'lower_body',
        4: 'other', 
        5: 'shoes',
        6: 'upper_body'
    },

    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
    
}