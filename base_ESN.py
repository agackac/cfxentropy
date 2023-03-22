def base_ESN(duration=500, bn=False, rnn_dims=128, radius=.2, kernel_size=5, maxpool=50, mlp_units=[64], use_mask=False, dropout=.5, nfreq=0, ntime=0):
    
    inputs = Input((duration,40))
    if use_mask==True:  # set to False by default as all recordings are above 5s duration - set to True for longer duration runs
        masking = Masking(0.)
        mask = masking.compute_mask(inputs)
        x = masking(inputs)
    else:
        mask = None
        x = inputs
        
    if (nfreq+ntime)>0:
        x = SpecAugment(freq_mask_param=nfreq, time_mask_param=ntime, n_freq_mask=nfreq, n_time_mask=ntime)(x)
        
    x = UnitNormalization()(x) if bn==False else BatchNormalization()(x)
    x = tfa.layers.ESN(rnn_dims, spectral_radius=radius, return_sequences=True)(x) 
    x = DepthwiseConv1D(kernel_size, padding='same')(x)
    x = PReLU('ones',shared_axes=[1])(x)
    
    if maxpool!=None:
        mp = MaxPooling1D(maxpool,1,padding='same')(x)
        x = Concatenate()([x,mp])
        
    x = GlobalAveragePooling1D()(x,mask=mask)
    
    for units in mlp_units:
        x = Dropout(dropout)(x)
        x = Dense(units,'relu')(x)
        
    x = Dense(1)(x)
    
    FE = Model(inputs,x)
    
    return FE
    
