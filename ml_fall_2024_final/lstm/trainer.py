def train_model(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_cont_batch, X_cat_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_cont_batch, X_cat_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}")
