class Config:
    class TRAIN:
        class OPTIM:
            BATCH_SIZE: int = 32
            BASE_EPOCHS: int = 80
            PRIOR_EPOCHS: int = 20
            BASE_LR: float = 1e-4
            PRIOR_LR: float = 1.5e-4
            BACKBONE_LR_SCALE: float = 0.35
            HEAD_LR_SCALE: float = 1.0
            WEIGHT_DECAY: float = 1e-4
            ETA_MIN: float = 1e-6
            PATIENCE: int = 10
            BASE_PATIENCE: int = 10
            PRIOR_PATIENCE: int = 6
            WARMUP_EPOCHS: int = 4
            GRAD_CLIP_NORM: float = 1.0

        class EVAL:
            USE_MODEL_EMA: bool = True
            MODEL_EMA_DECAY: float = 0.999
            MODEL_EMA_START_EPOCH: int = 1
            EMA_USE_FOR_EVAL: bool = True

    class MODEL:
        class BASE:
            MODEL_TYPE: str = "benchmark"
            ADAPTER_MODULE: str = ""
            ADAPTER_CLASS: str = ""
            EXTERNAL_CODE_ROOT: str = ""
            EXTERNAL_CHECKPOINT: str = ""
            EXTERNAL_CONFIG: str = ""
            EXTERNAL_TRAIN_CONFIG: str = ""
            EXTERNAL_DATA_ROOT: str = ""
            EXTERNAL_TOKENIZER: str = ""
            PRIOR_BRANCH_ENABLED: bool = True
            PRIOR_AWARE_CONDITIONING_ENABLED: bool = True
            LEARNED_EVIDENCE_COMPRESSION_ENABLED: bool = True
            COMPRESSED_TOKEN_INTERACTION_ENABLED: bool = True
            RANK_LOSS_WEIGHT: float = 0.0
            RANK_LOSS_MARGIN: float = 0.20
            RANK_LOSS_TEMPERATURE: float = 1.0
            CONSENSUS_ENABLED: bool = False
            CONSENSUS_VIEW_DROPOUT: float = 0.05
            CONSENSUS_AGREEMENT_WEIGHT: float = 0.0

        class BACKBONE:
            MACRO_DRUG_INPUT_DIM: int = 768
            MACRO_TARGET_INPUT_DIM: int = 1280
            DRUG_TOKEN_DIM: int = 768
            PROT_TOKEN_DIM: int = 1280
            HIDDEN_DIM: int = 512
            DROPOUT: float = 0.12
            CAN_NUM_HEADS: int = 8
            CAN_NUM_LAYERS: int = 2
            CAN_GROUP_SIZE: int = 2

    class DATA:
        class PATHS:
            ROOT_DIR: str = "./datasets"

        class ENVIRONMENT:
            PSEUDO_ENV_MODE: str = "pair_macro"
            PSEUDO_ENV_BUCKETS: int = 8
            PSEUDO_ENV_SEED: int = 2026

        class LOADER:
            NUM_WORKERS: int = 4
            PERSISTENT_WORKERS: bool = True
            PREFETCH_FACTOR: int = 2

        class FILES:
            FILE_DRUG_MACRO: str = "drug_cls_feat.pkl"
            FILE_DRUG_TOKEN: str = "drug_token_feat.pkl"
            FILE_PROT_MACRO: str = "prot_cls_feat.pkl"
            FILE_PROT_TOKEN: str = "prot_token_feat.pkl"
