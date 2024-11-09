from transformers import TrainingArguments
from intrinsic_loss import IntrinsicLoss
from llama_model import CuriousLLaMModel

# Create an instance of the IntrinsicLoss class
intrinsic_loss = IntrinsicLoss(lambda_val=0.01)

# Create an instance of the CuriousLLaMModel class
modified_model = CuriousLLaMModel.from_pretrained('llamalab/llama-base')

# Set up training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=1000,
    weight_decay=0.01,
)

# Move the model to the GPU (if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
modified_model.to(device)

# Load your dataset and prepare it for training
#dataset = YourDataset() ## need to find one lol
train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_train_batch_size, shuffle=True)
eval_dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

# Train the modified model
modified_model.train()
for epoch in range(training_args.num_train_epochs):
    for batch in train_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Compute the intrinsic loss
        output = modified_model(input_ids, attention_mask=attention_mask, labels=labels)
        original_loss = nn.CrossEntropyLoss()(output, labels)
        reward_loss = intrinsic_loss(output, labels).item()
        loss = reward_loss + original_loss
        
        # Backpropagate and update the model parameters
        loss.backward()
        optimizer.step()

    # Evaluate the model on the validation set
    modified_model.eval()
    eval_loss = 0
    for batch in eval_dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        output = modified_model(input_ids, attention_mask=attention_mask, labels=labels)
        original_loss = nn.CrossEntropyLoss()(output, labels)
        reward_loss = intrinsic_loss(output, labels)
        loss = original_loss + reward_loss
        
        eval_loss += loss.item()

    print(f'Epoch {epoch+1}, Eval Loss: {eval_loss / len(eval_dataloader)}')
    



