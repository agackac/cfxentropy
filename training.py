def make_cf(raw=False, center=False, multitask=False):
    return Siamese(FE, naive=False, raw=raw, center=center, add_loss=multitask)
    
def make_naive(FE, cutoff=10):
    return Siamese(FE, naive=True, cutoff=cutoff)
                 
def run_trial(siam, train, val, test=[], duration=500, epochs=20, spe=100, val_steps=10, patience=None, restore=False, clip=None):

    siam.compile(optimizer = Adam(1e-3, clipnorm=clip), 
                 weighted_metrics = []) # to prevent Keras backend from thinking passed tuples are class weights 
    
    callbacks = [EarlyStopping(patience=patience, restore_best_weights=restore)] if patience!=None else None
    history   = siam.fit(train, validation_data=val, epochs=epochs, steps_per_epoch=spe, validation_steps=val_steps, callbacks=callbacks)
    
    test_results = [siam.evaluate(test_set) for test_set in test]
    
    return siam, history, test_results
