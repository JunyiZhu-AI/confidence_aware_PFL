import json
import click
from vem import train


@click.command()
@click.option(
    "--dataset",
    default="CIFAR10",
    type=click.Choice(["CIFAR10", "CIFAR100"], case_sensitive=False),
)
@click.option(
    "--model",
    default="CNNCifar",
    type=click.Choice(["CNNCifar", "CNNCifar100"], case_sensitive=False),
)
@click.option("--batch_size", default=10000, help="Batch size.")
@click.option("--lr_head", default=0.0003, help="Lr for base model.")
@click.option("--lr_base", default=0.0001, help="Lr for local model.")
@click.option("--momentum", default=0.9, help="Momentum for optimizer.")
@click.option("--head_epochs", default=10, help="Number of epochs for head network training.")
@click.option("--base_epochs", default=10, help="Number of epochs for base network training.")
@click.option("--n_mc", default=5, help="number of classes in each client.")
@click.option("--scale", default=1, help="Initialized scale of Gaussian posterior.")
@click.option("--n_labels", default=2, help="Number of classes in each local dataset.")
@click.option("--relabel", is_flag=True)
@click.option("--n_rounds", default=100, help="Number of communication rounds.")
@click.option("--max_data", default=0, help="The number of data points for the overall dataset.")
@click.option("--n_clients", default=500, help="Number of clients.")
@click.option("--sampling_rate", default=0.1, help="Clients sampling rate.")
@click.option("--path_to_data", default="./data")
@click.option("--seed", default=0, help="Random seed.")
@click.option(
    "--config", help="Path to the configuration file.", default=None,
)
def main(**kwargs):
    if kwargs["config"]:
        with open(kwargs["config"]) as f:
            kwargs = json.load(f)
    else:
        del kwargs["config"]

    print(kwargs)
    train(**kwargs)


if __name__ == "__main__":
    main()
