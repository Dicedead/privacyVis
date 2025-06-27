# Privacy Visualisation through Hypothesis Testing

**EPFL Master's thesis: Salim Najib, supervised by Prof. Yanina Shkel and Cemre Çadir.**

## Abstract
Differential privacy has stood the test of time as a measure of information leakage, in a context where the privacy of individuals has become an increasingly critical stake. To help non-privacy experts get familiar with some of its recent generalisations and theoretical results on composition, we devise a visualisation tool for differential privacy, leveraging its binary hypothesis testing interpretation. It aims to show multiple kinds of privacy regions, to showcase differences between composition theorems, and to aid in gaining an intuition on privacy-utility trade-offs for some common mechanisms.

## How to run this tool
Once the required libraries in ``requirements.txt`` have been installed, run the Python file ``src/main.py``. 

## How to use this tool
When starting the software, a main menu prompts a choice between the two following types of windows:

- A privacy regions window, depicting the privacy regions of differential privacy, some of its generalisations, and some results on the composition of differentially private mechanisms.
- A utility-privacy trade-off window for specific combinations of query, utility, and mechanism. It shows how the selected utility evolves as a function of the parameters of the privacy-preserving mechanism, along with the corresponding privacy region.

More info, including on the theory, in the report: ``master_thesis.pdf``.