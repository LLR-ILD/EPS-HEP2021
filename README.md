# EPS-HEP2021

Build the presentation given online at the
[EPS-HEP2021](https://indico.desy.de/event/28202/contributions/106023/)
conference.

## Fast access

Several assets can be directly accessed without running the code:

- [**Download**](https://github.com/LLR-ILD/EPS-HEP2021/releases/download/v1.0/presentation.pdf)
  the pdf presentation given at EPS-HEP2021.
- [**Access**](https://github.com/LLR-ILD/EPS-HEP2021/tree/gh-action-result/make-all/build)
  the _current_ version of the documents from this repository.

In all cases you might alternatively want to navugate to the parent of the link
to view byproducts of the artifact creation, including the raw figures.

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
