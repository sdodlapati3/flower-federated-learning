"""flower-tutorial: A Flower / PyTorch app.

Part 4: Communicate Custom Messages
- Send arbitrary Python objects (TrainProcessMetadata) from ClientApp to ServerApp
- Serialize using pickle and send via ConfigRecord
- Track detailed training metrics per client
"""

import pickle
import time

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from flower_tutorial.task import Net, TrainProcessMetadata, load_data
from flower_tutorial.task import test as test_fn

# Flower ClientApp
app = ClientApp()


def train_with_metadata(net, trainloader, epochs, lr, device):
    """Train the model and return detailed metadata about the training process.
    
    This enhanced training function captures:
    - Training time
    - Loss per epoch
    - Convergence status
    - Number of batches processed
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    
    start_time = time.time()
    training_losses = {}
    num_batches = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        for images, labels in trainloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batch_count += 1
            num_batches += 1
        
        avg_epoch_loss = epoch_loss / batch_count
        training_losses[f"epoch_{epoch + 1}"] = avg_epoch_loss
    
    end_time = time.time()
    training_time = end_time - start_time
    
    # Get final loss (last epoch's average)
    final_loss = list(training_losses.values())[-1]
    
    # Consider "converged" if final loss is below threshold
    converged = final_loss < 1.5
    
    # Create metadata object
    metadata = TrainProcessMetadata(
        training_time=training_time,
        converged=converged,
        training_losses=training_losses,
        final_loss=final_loss,
        num_batches=num_batches,
    )
    
    return final_loss, metadata


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data and send custom metadata."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, num_partitions)

    # Call the enhanced training function that returns metadata
    train_loss, train_metadata = train_with_metadata(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Serialize the TrainProcessMetadata object to bytes
    train_meta_bytes = pickle.dumps(train_metadata)
    
    # Construct ConfigRecord with serialized metadata
    config_record = ConfigRecord({"meta": train_meta_bytes})

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
        "training_time": train_metadata.training_time,  # Also send as metric for aggregation
    }
    metric_record = MetricRecord(metrics)
    
    # Include the serialized metadata in the content
    content = RecordDict({
        "arrays": model_record,
        "metrics": metric_record,
        "train_metadata": config_record,  # Custom serialized object
    })
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, num_partitions)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
