import torch
import numpy as np

TRAINING_MESSAGE = "Epoch: {}/{}. Train set: Average loss: {:.4f}"
VALIDATION_MESSAGE = "\nEpoch: {}/{}. Validation set: Average loss: {:.4f}"
METRIC_MESSAGE = "\t{}: {}"
BATCH_MESSAGE = "Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}"


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
        scheduler.step()

        # Train stage
        train_loss, metrics = train_epoch(
            train_loader, model, loss_fn, optimizer, cuda, log_interval, metrics
        )
        message = TRAINING_MESSAGE.format(epoch + 1, n_epochs, train_loss)
        for metric in metrics:
            message += METRIC_MESSAGE.format(metric.name(), metric.value())

        # Validation stage
        val_loss, metrics = test_epoch(val_loader, model, loss_fn, cuda, metrics)
        val_loss /= len(val_loader)
        message += VALIDATION_MESSAGE.format(epoch + 1, n_epochs, val_loss)

        for metric in metrics:
            message += METRIC_MESSAGE.format(metric.name(), metric.value())

        print(message)


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

    for batch_idx, (data, target) in enumerate(train_loader):
        target = target if len(target) > 0 else None
        if not type(data) in (tuple, list):
            data = (data,)
        if cuda:
            data = tuple(d.cuda() for d in data)
            if target is not None:
                target = target.cuda()

        prova_len = len(data[0])
        for feature_idx in range(prova_len):
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
            message = BATCH_MESSAGE.format(
                batch_idx * len(data[0]),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                np.mean(losses)
            )
            for metric in metrics:
                message += METRIC_MESSAGE.format(metric.name(), metric.value())

            print(message)
            losses = []

    total_loss /= (total_batches + 1)

    return total_loss, metrics


def test_epoch(val_loader, model, loss_fn, cuda, metrics):
    with torch.no_grad():
        for metric in metrics:
            metric.reset()
        model.eval()
        val_loss = 0
        for batch_idx, (data, target) in enumerate(val_loader):
            target = target if len(target) > 0 else None
            if not type(data) in (tuple, list):
                data = (data,)
            if cuda:
                data = tuple(d.cuda() for d in data)
                if target is not None:
                    target = target.cuda()

            outputs = model(*data)

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

            for metric in metrics:
                metric(outputs, target, loss_outputs)

    return val_loss, metrics
