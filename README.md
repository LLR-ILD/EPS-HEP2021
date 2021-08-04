# EPS-HEP2021

Build the presentation given online at the
[EPS-HEP2021](https://indico.desy.de/event/28202/contributions/106023/)
conference.

## Usage

Should be as simple as

```bash
source init.sh  # Always necessary, but should be fast after the first time.
make
```

Some targets are provided for covenience
(e.g. `make fit` to recreate the fit images without rebuilding the presentation).
If needed, they can be found in `Makefile`.

## Building new tables

To change the tables programatically (define different categories),
you need access to the pre-event root files.
Some paths for obtaining them are are outlined in `init.sh` and can be steered
through arguments for this script.

The [`higgstables-config.yaml`](code/data/higgstables-config.yaml)
file gives an overview of what can be changed.
This includes defining different categories
or changing the event selection step.
