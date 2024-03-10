from box import Box

config = Box({'seed': 2021,
          'root': '/home/olga/Programs/mlops-hw/data', 
          'n_splits': 5,
          'epoch': 20,
          'trainer': {
              'accumulate_grad_batches': 1,
              'fast_dev_run': False,
              'num_sanity_val_steps': 0,
          },
          'transform':{
              'name': 'get_default_transforms',
              'image_size': 224
          },
          'train_loader':{
              'batch_size': 64,
              'shuffle': True,
              'num_workers': 4,
              'pin_memory': False,
              'drop_last': True,
          },
          'val_loader': {
              'batch_size': 64,
              'shuffle': False,
              'num_workers': 4,
              'pin_memory': False,
              'drop_last': False
         },
          'model':{
              'name': 'swin_tiny_patch4_window7_224',
              'output_dim': 1
          },
          'optimizer':{
              'name': 'optim.AdamW',
              'params':{
                  'lr': 1e-5
              },
          },
          'scheduler':{
              'name': 'optim.lr_scheduler.CosineAnnealingWarmRestarts',
              'params':{
                  'T_0': 20,
                  'eta_min': 1e-4,
              }
          },
          'loss': 'nn.BCEWithLogitsLoss',
})
