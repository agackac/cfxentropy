def make_ds(df, batch_size=128, timesteps=500, reps=-1, start='random', ws=True):
    
    if ws==True:
        x_cols = ['D','H']
        i_cols = ['y'] 
        y_cols = ['gender']
    elif ws=='all':
        
        x_cols = ['librosa_mel']
        i_cols = []
        y_cols = []
    else:
        x_cols = ['librosa_mel']
        i_cols = []
        y_cols = ['PHQ8_total' if 'PHQ8_total' in df.columns else 'T0_total']
        
    features = 40
    np.random.seed(0)
    tf.random.set_seed(0)
    
    def map_func(path, slices=timesteps, start=start):
        rec = np.load(path,allow_pickle=True).T 
        excess = rec.shape[0]-slices
        if excess>0:
            start = 0 if start=='zero' else np.random.choice(range(excess))
            out   = rec[start:start+slices,:,]
        else:
            out   = np.zeros((slices, rec.shape[1]))
            out[:rec.shape[0],:] = rec
        return out.astype('float32')
    
    @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
    def tf_load_rec(input):
        rec = tf.numpy_function(map_func, [input], tf.float32)
        rec.set_shape((timesteps,features))
        return rec
    
        
    ds = []
    
    for col in x_cols:
        x = df[col].values
        x = Dataset.from_tensor_slices(x).map(tf_load_rec)
        x = [x]
        for i in i_cols:
            y = df[col+i].values
            y = np.reshape(y,(-1,1))
            y = Dataset.from_tensor_slices(y).map(lambda x: tf.cast(x,tf.float32))
            x.append(y)
        x = Dataset.zip(tuple(x))

        ds.append(x)
        
        
    for col in y_cols:
        y = df[col].values
        y = np.reshape(y,(-1,1))
        y = Dataset.from_tensor_slices(y).map(lambda x: tf.cast(x,tf.float32))
        ds.append(y)
        
        
    ds = Dataset.zip(tuple(ds))
        
    if reps==-1:
        ds = ds.repeat()
    else:
        ds = ds.repeat(reps)
    
    ds = ds.shuffle(buffer_size=len(df),seed=1)
    
    if batch_size!=None:
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(8)
    
    return ds

# For concurrent tracking of generalization on counterfactual pairs vs. random cross-sectional sample
def make_dual_val(paired, naive, batch_size=128, timesteps=500, start='random'):
    
    paired = make_ds(paired, batch_size=None, timesteps=timesteps, start=start)
    naive  = make_ds(naive, batch_size=None, timesteps=timesteps, start=start, ws=False)
    
    ds = Dataset.zip(tuple([paired,naive]))
    ds = ds.batch(batch_size).prefetch(8)
    return ds


# Makes batches containing random slices of a single file (predictions are averaged by the Siamese wrapper)
def make_eval(x, reps=100, timesteps=500, start='random'):
    features=40
    def map_func(path, slices=timesteps, start=start):
        rec = np.load(path,allow_pickle=True).T 
        excess = rec.shape[0]-slices
        if excess>0:
            start = 0 if start=='zero' else np.random.choice(range(excess))
            out   = rec[start:start+slices,:,]
        else:
            out   = np.zeros((slices, rec.shape[1]))
            out[:rec.shape[0],:] = rec
        return out.astype('float32')
    
    @tf.function(input_signature=[tf.TensorSpec(None, tf.string)])
    def tf_load_rec(input):
        rec = tf.numpy_function(map_func, [input], tf.float32)
        rec.set_shape((timesteps,features))
        return rec
    
    ds = Dataset.from_tensor_slices([x[0]]).repeat(reps)
    for i in x[1:]:
        ds = ds.concatenate(Dataset.from_tensor_slices([i]).repeat(reps))
    ds = ds.map(tf_load_rec)
    ds = ds.batch(reps)
    return ds
