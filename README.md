SpikingRx-on-OAI
├── archive
│   ├── checkpoints
│   │   └── best_spikingrx_model.BACKUP_20260209_161720.pth
│   └── legacy_debug
│       ├── inference
│       ├── tools
│       └── train
│
├── docs
│   ├── images
│   │   ├── gnB_UE connect.png
│   │   ├── damp from nr_dlsch_demodulation.png
│   │   └── model_pictures
│   │
│   ├── notes
│   │   ├── 0212_log.md
│   │   ├── 0216_log.md
│   │   ├── 0218_log.md
│   │   ├── 0220_log.md
│   │   ├── 0223_log.md
│   │   ├── 0224_log.md
│   │   ├── 0301_log.md
│   │   ├── 0304_log.md
│   │   ├── 0305_log.md
│   │   ├── 0306_log.md
│   │   ├── 0308_log.md
│   │   ├── 0309_log.md
│   │   └── 0314_log.md
│   │
│   ├── papers
│   └── results
│       ├── inference
│       ├── train
│       └── visualize
│
├── oai_change
│   ├── openair1
│   │   └── PHY
│   │       ├── CODING
│   │       ├── NR_TRANSPORT
│   │       └── NR_UE_TRANSPORT
│   │
│   ├── openair2
│   │   └── LAYER2
│   │
│   ├── radio
│   │   └── rfsimulator
│   │
│   └── targets
│
├── src
│   ├── data
│   │   ├── dataset_oai_bundle.py
│   │   └── oai_to_spikingrx_tensor.py
│   │
│   ├── inference
│   │   ├── batch_inference_on_bundle.py
│   │   ├── check_oai_llr_decode.py
│   │   └── run_spikingrx_on_oai_dump.py
│   │
│   ├── models
│   │   ├── conv_block.py
│   │   ├── lif_neuron.py
│   │   ├── norm_layer.py
│   │   ├── sew_block.py
│   │   └── spikingrx_model.py
│   │
│   ├── tools
│   │   ├── bundle_records.py
│   │   └── tests
│   │
│   ├── train
│   │   └── train_spikingrx_oai.py
│   │
│   └── visualize
│       └── visualize_spiking_activity.py
│
├── oai_snapshot.bundle
└── README.md
