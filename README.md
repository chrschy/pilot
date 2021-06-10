# PILOT: Introducing Transformers for Probabilistic Sound Event Localization

This repository contains the codebase accompanying our publication:

> Christopher Schymura, Benedikt Bönninghoff, Tsubasa Ochiai, Marc Delcroix, Keisuke Kinoshita, Tomohiro Nakatani, Shoko Araki, Dorothea Kolossa, "PILOT: Introducing Transformers for Probabilistic Sound Event Localization", *INTERSPEECH 2021*

[ [arXiv](https://arxiv.org/abs/2106.03903) ]

## 📓 Summary

Sound event localization aims at estimating the positions of sound sources in the environment with respect to an acoustic receiver (e.g. a microphone array). Recent advances in this domain most prominently focused on utilizing deep recurrent neural networks. Inspired by the success of transformer architectures as a suitable alternative to classical recurrent neural networks, the PILOT (*P*robab*i*listic *L*ocalization of S*o*unds with *T*ransformers) model is a transformer-based sound event localization framework, where temporal dependencies in the received multi-channel audio signals are captured via self-attention mechanisms. Additionally, the estimated sound event positions are represented as multivariate Gaussian variables, yielding an additional notion of uncertainty, which many previously proposed deep learning-based systems designed for this application do not provide.

## 🚀 Getting started

You can train and evaluate the PILOT model using the [ANSIM](https://doi.org/10.5281/zenodo.1237703), [RESIM](https://doi.org/10.5281/zenodo.1237707) and [REAL](https://doi.org/10.5281/zenodo.1237793) sound event localization and detection datasets. We have prepared an 
