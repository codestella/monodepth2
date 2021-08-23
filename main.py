from __future__ import absolute_import, division, print_function

from tum_trainer import TUM_Trainer
from tum_options import TUM_Options

options = TUM_Options()
opts = options.parse()

if __name__ == "__main__":
    trainer = TUM_Trainer(opts)
    trainer.train()
