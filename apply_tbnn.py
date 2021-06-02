from tbnns.tbnn import TBNN

def trainNetwork(x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
                 loss_weight_train=None, loss_weight_dev=None):
    """
    This function takes in training data and validation data (aka dev set)
    and runs the training routine. We initialize parameters of the TBNN-s 
    through the dictionary FLAGS and call nn.train() for training.
    """
        
    # Flags indicating parameters of the TBNN. This is a 
    # comprehensive list, with all flags that can be prescribed
    FLAGS = {} # FLAGS is a dictionary with the following keys:
    
    # num features used to be 7
    FLAGS['num_features'] = 3 # number of features to be used
    FLAGS['num_basis'] = 10 # number of tensor basis in the expansion
    FLAGS['num_layers'] = 25 # number of hidden layers 8 
    FLAGS['num_neurons'] = 100 # number of hidden units in each layer 30 
    
    FLAGS['learning_rate'] = 5.0e-5 # learning rate for SGD algorithm
    FLAGS['num_epochs'] = 2000 # maximum number of epochs to run
    FLAGS['early_stop_dev'] = 20 # after this many evaluations without improvement, stop training           
    FLAGS['eval_every'] = 100 # when to evaluate losses
    FLAGS['train_batch_size'] = 40 # number of points per batch
    
    FLAGS['loss_type'] = 'l1' # loss type    
    FLAGS['c_reg'] = 1e-7 # L2 regularization strength   
    FLAGS['drop_prob'] = 0.0 # dropout probability at training time   
     
    # Initialize TBNN with given FLAGS
    nn = TBNN()
    nn.initializeGraph(FLAGS)   
    
    # Path to write TBNN metadata and parameters
    path_params = 'checkpoints/nn_test_tbnn.ckpt'
    path_class = 'nn_test_tbnn.pckl'
    
    # Train and save to disk
    best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list = nn.train(path_params,
             x_train, tb_train, b_train, x_dev, tb_dev, b_dev,
             train_loss_weight=loss_weight_train, dev_loss_weight=loss_weight_dev,
             detailed_losses=True)
    nn.saveToDisk("Testing TBNN", path_class)

    return best_dev_loss, end_dev_loss, step_list, train_loss_list, dev_loss_list
  

def applyNetwork(x_test, tb_test, b_test, gradu_test, nut_test, tke_test):
    """
    This function takes in test data and applies previously trained TBNN.    
    """    
    
    # ----- Load meta-data and initialize parameters
    # path_class should correspond to the path on disk of an existing trained network 
    path_class = 'nn_test_tbnn.pckl' 
    nn = TBNN()
    nn.loadFromDisk(path_class, verbose=True)
    
    # Apply TBNN on the test set to get losses
    loss, loss_pred, _, _ = nn.getTotalLosses(x_test, tb_test, b_test)
    loss_pred_rans = nn.getRansLoss(b_test, gradu_test, tke_test, nut_test)   
    print("JICF r=1 LEVM loss: {:g}".format(loss_pred_rans))
    print("JICF r=1 TBNN-s loss: {:g}".format(loss_pred))
    print("")
    
    # Now, apply TBNN on the test set to get a predicted anisotropy matrix
    # This can be used in a RANS solver to get improved velocity field predictions
    b_pred, g = nn.getTotalAnisotropy(x_test, tb_test)    
    print("Predicted anisotropy shape: ", b_pred.shape)
    print("Predicted coefficients g_n shape: ", g.shape)

    return b_pred, g
    


