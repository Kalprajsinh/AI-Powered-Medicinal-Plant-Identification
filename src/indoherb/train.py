class Trainer:
    def __init__(self, model, model_name):
        self.model = model.model
        self.model_name = model_name
        self.device = model.device

        # history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'test_loss': [], 'test_acc': [],
            'precision': [], 'recall': [], 'f1': []
        }

    def train_epoch(self, train_loader, criterion, optimizer):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f'{self.model_name} Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.3f}',
                'Acc': f'{100.*correct/total:.1f}%'
            })

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def evaluate(self, test_loader, criterion):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        epoch_loss = running_loss / len(test_loader)
        epoch_acc = 100. * correct / total

        precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)

        return epoch_loss, epoch_acc, precision, recall, f1

    def train(self, train_loader, test_loader, epochs=50, lr=0.001, gamma=0.9):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # Learning Rate Scheduler
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

        best_acc = 0.0

        print(f"\ntraining {self.model_name}")
        print(f"   Epochs: {epochs}, Learning Rate: {lr}, Gamma: {gamma}")

        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)

            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)

            # Evaluation phase
            test_loss, test_acc, precision, recall, f1 = self.evaluate(test_loader, criterion)

            # KEY DIFFERENCE: Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]

            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)

            # Print progress
            print(f'Precision:  {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Test Loss:  {test_loss:.4f}, Test Acc:  {test_acc:.2f}%')

            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), f'{self.model_name}_best.pth')

        # Load best model for final use
        self.model.load_state_dict(torch.load(f'{self.model_name}_best.pth'))
        print(f'\nTraining completed \n Accuracy: {best_acc:.2f}%')

        return best_acc

print("DataLoader")

BATCH_SIZE = 32

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

num_classes = len(train_dataset.classes)
print(f"\n🌿 Number of plant classes: {num_classes}")

print("🤖 Initializing models...")
resnet_model = PlantModel("resnet34", num_classes)
densenet_model = PlantModel("densenet121", num_classes)

resnet_trainer = Trainer(resnet_model, "ResNet34")
densenet_trainer = Trainer(densenet_model, "DenseNet121")

# ResNet34
print("\n" + "="*60)
resnet_best_acc = resnet_trainer.train(
    train_loader, test_loader,
    epochs=50, lr=0.001, gamma=0.9
)

# DenseNet121
print("\n" + "="*60)
densenet_best_acc = densenet_trainer.train(
    train_loader, test_loader,
    epochs=50, lr=0.001, gamma=0.9
)
