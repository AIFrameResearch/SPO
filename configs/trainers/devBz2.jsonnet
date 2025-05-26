{
    trainer+: {
        general_training_args+: {
            per_device_train_batch_size: 2,
            gradient_accumulation_steps: null,  // Will be auto computed
        },
    },
}
+ (import 'lam0.96.jsonnet')
