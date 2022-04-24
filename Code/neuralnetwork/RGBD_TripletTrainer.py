import torch
import numpy as np

TRAINING_MESSAGE = "Epoch: {}/{}. Train set: Average loss: {:.4f}"
VALIDATION_MESSAGE = "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}"
METRIC_MESSAGE = "\t{}: {}"
TRAIN_MESSAGE = "Train: [Scene: {}/{} (Current: #{}); Batch: {}/{} -> " \
                "({:.0f}%)]\t-\tLoss: {:.6f}"
VAL_MESSAGE = "Val: [Scene: {}/{} (Current: #{}); Batch: {}/{} -> " \
                "({:.0f}%)]\t-\tLoss: {:.6f}"
EPOCH_MESSAGE = "Epoch {}/{}"


def fit(
    train_loader,
    val_loader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    n_epochs,
    cuda,
    log_interval,
    metrics=None,
    start_epoch=0
):
    """
    Loaders, model, loss function and metrics should work together for
    a given task,i.e. The model should be able to process data output of
    loaders, loss function should process target output of loaders and outputs
    from the model
    """
    if metrics is None:
        metrics = []

    for epoch in range(0, start_epoch):
        scheduler.step()

    for epoch in range(start_epoch, n_epochs):
        print("\n#############################################################")
        message = EPOCH_MESSAGE.format(
            epoch,
            n_epochs - start_epoch
        )
        print(message)

        # Train stage
        print("\nTraining:")
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics
        )

        scheduler.step()

        message = TRAINING_MESSAGE.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += METRIC_MESSAGE.format(metric.name(), metric.value())

        print("\nValidation:")

        # Validation stage
        val_loss, metrics = val_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        message += VALIDATION_MESSAGE.format(epoch + 1, n_epochs, val_loss)

        for metric in metrics:
            message += METRIC_MESSAGE.format(metric.name(), metric.value())

        print("\n")
        print(message)

    print("---###@@@!!!! MODEL FIT COMPLETED !!!@@@###---")


def train_epoch(
    train_loader,
    model,
    loss_fn,
    optimizer,
    cuda,
    log_interval,
    metrics
):
    for metric in metrics:
        metric.reset()

    model.train()
    losses = []
    total_loss = 0
    total_batches = 0
    total_scenes = train_loader.dataset.get_num_scenes()

    for scene in range(total_scenes):
        for batch_idx, (data, target) in enumerate(train_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            for feature_idx in range(len(data[0])):
                anchor = data[0][feature_idx].float()
                pos = data[1][feature_idx].float()
                neg = data[2][feature_idx].float()

                optimizer.zero_grad()
                outputs = model(anchor, pos, neg)

                if type(outputs) not in (tuple, list):
                    outputs = (outputs,)

                loss_inputs = outputs
                if target is not None:
                    target = (target,)
                    loss_inputs += target

                loss_outputs = loss_fn(*loss_inputs)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else\
                    loss_outputs
                losses.append(loss.item())
                total_loss += loss.item()
                total_batches += 1
                loss.backward()
                optimizer.step()

                for metric in metrics:
                    metric(outputs, target, loss_outputs)

            if batch_idx % log_interval == 0:
                message = TRAIN_MESSAGE.format(
                    scene + 1,
                    total_scenes,
                    train_loader.dataset.batch_scene_idx,
                    batch_idx * len(data[0]),
                    len(train_loader) * len(data[0]),
                    100. * batch_idx / len(train_loader),
                    np.mean(losses)
                )
                for metric in metrics:
                    message += METRIC_MESSAGE.format(
                        metric.name(),
                        metric.value()
                    )

                print(message)
                losses = []
        print("---")

    total_loss /= (total_batches + 1)

    return total_loss, metrics


def val_epoch(
    val_loader,
    model,
    loss_fn,
    cuda,
    metrics,
    log_interval=1
):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        losses = []
        val_loss = 0
        total_batches = 0
        total_scenes = val_loader.dataset.get_num_scenes()

        for scene in range(total_scenes):
            for batch_idx, (data, target) in enumerate(val_loader):
                target = target if len(target) > 0 else None
                if not type(data) in (tuple, list):
                    data = (data,)
                if cuda:
                    data = tuple(d.cuda() for d in data)
                    if target is not None:
                        target = target.cuda()

                for feature_idx in range(len(data[0])):
                    anchor = data[0][feature_idx].float()
                    pos = data[1][feature_idx].float()
                    neg = data[2][feature_idx].float()

                    outputs = model(anchor, pos, neg)

                    if type(outputs) not in (tuple, list):
                        outputs = (outputs,)
                    loss_inputs = outputs
                    if target is not None:
                        target = (target,)
                        loss_inputs += target

                    loss_outputs = loss_fn(*loss_inputs)
                    loss = loss_outputs[0] if type(loss_outputs) \
                                              in (tuple, list) else loss_outputs
                    val_loss += loss.item()
                    losses.append(loss.item())
                    total_batches += 1

                    for metric in metrics:
                        metric(outputs, target, loss_outputs)

                if batch_idx % log_interval == 0:
                    message = VAL_MESSAGE.format(
                        scene + 1,
                        total_scenes,
                        val_loader.dataset.batch_scene_idx,
                        batch_idx * len(data[0]),
                        len(val_loader) * len(data[0]),
                        100. * batch_idx / len(val_loader),
                        np.mean(losses)
                    )
                    for metric in metrics:
                        message += METRIC_MESSAGE.format(
                            metric.name(),
                            metric.value()
                        )

                    print(message)
                    losses = []

    val_loss /= (total_batches + 1)

    return val_loss, metrics
