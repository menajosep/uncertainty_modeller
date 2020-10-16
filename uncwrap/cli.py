"""A sample CLI."""

import click
import log

from . import UncertaintyWrapperEstimator


@click.command()
@click.argument('feet')
def main(feet=None):
    log.init()

    wrapper = UncertaintyWrapperEstimator()


if __name__ == '__main__':  # pragma: no cover
    main()
