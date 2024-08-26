import torch

def transformer_temporal_softmax_loss(x, y, mask):
    """
    Computes the temporal softmax loss for a batch of predictions.

    This loss function is designed for sequence models where each time step has a prediction.
    It computes the cross-entropy loss for each time step and then averages the loss over
    the valid entries as indicated by the mask.

    Arguments:
        x (torch.Tensor): Predicted scores of shape (N, T, V), where N is the batch size,
                          T is the sequence length, and V is the vocabulary size.
        y (torch.Tensor): Ground-truth indices of shape (N, T).
        mask (torch.Tensor): Boolean mask of shape (N, T) where mask[i, t] indicates whether
                             the output at x[i, t] should contribute to the loss.

    Returns:
        torch.Tensor: Computed loss value (scalar).
    """
    
    N, T, V = x.shape

    x_flat = x.reshape(N * T, V)
    y_flat = y.reshape(N * T)
    mask_flat = mask.reshape(N * T)

    loss = torch.nn.functional.cross_entropy(x_flat, y_flat, reduction='none')
    loss = loss * mask_flat
    loss = loss.sum() / mask_flat.sum()  # Normalize by the number of contributing elements

    return loss

def train_model(model, train_data_loader, val_data_loader, optimizer, scheduler,  num_epochs=10):
    """
    Trains the model and evaluates it on validation data.

    This function handles the training and validation of a PyTorch model for a specified number of epochs.
    It also manages the learning rate scheduling and saves the model after each epoch.

    Arguments:
        model (torch.nn.Module): The model to be trained.
        train_data_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        num_epochs (int): Number of epochs to train the model.

    Returns:
        None
    """
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        
        # Training loop
        for batch in train_data_loader:
            img_features, captions = batch
            img_features = img_features.to(device)
            captions = captions.to(device)
            
            optimizer.zero_grad()  # Clear gradients
            
            # Forward pass
            outputs = model(img_features, captions[:, :-1])  # Exclude the last token for inputs
            
            # Prepare target and mask
            captions_out = captions[:, 1:]  # Shift for target
            mask = captions_out != model._null  # Assuming model._null_token exists for padding

            # Calculate loss with mask
            loss = transformer_temporal_softmax_loss(outputs, captions_out, mask)
            loss.backward()  # Backward pass
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()  # Update parameters
            
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_data_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}")
        
        # Validation loop
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        with torch.no_grad():  # No need to track gradients during validation
            for batch in val_data_loader:
                img_features, captions = batch
                img_features = img_features.to(device)
                captions = captions.to(device)
                
                outputs = model(img_features, captions[:, :-1])
                
                # Prepare target and mask
                captions_out = captions[:, 1:]  # Shift for target
                mask = captions_out != model._null  # Assuming model._null_token exists for padding

                # Calculate loss with mask
                loss = transformer_temporal_softmax_loss(outputs, captions_out, mask)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_data_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Learning rate scheduler step (uncomment if needed)
        scheduler.step(avg_val_loss)

        # Save the model after each epoch
        torch.save(model.state_dict(), f'./save_model/Decoder_model_epoch_{epoch+1}.pth')
        