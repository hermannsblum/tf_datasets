<div itemscope itemtype="http://schema.org/Dataset">
  <div itemscope itemprop="includedInDataCatalog" itemtype="http://schema.org/DataCatalog">
    <meta itemprop="name" content="TensorFlow Datasets" />
  </div>

  <meta itemprop="name" content="big_patent" />
  <meta itemprop="description" content="BIGPATENT, consisting of 1.3 million records of U.S. patent documents&#10;along with human written abstractive summaries.&#10;Each US patent application is filed under a Cooperative Patent Classification&#10;(CPC) code. There are nine such classification categories:&#10;A (Human Necessities), B (Performing Operations; Transporting),&#10;C (Chemistry; Metallurgy), D (Textiles; Paper), E (Fixed Constructions),&#10;F (Mechanical Engineering; Lightning; Heating; Weapons; Blasting),&#10;G (Physics), H (Electricity), and&#10;Y (General tagging of new or cross-sectional technology)&#10;&#10;There are two features:&#10;  - description: detailed description of patent.&#10;  - summary: Patent abastract.&#10;&#10;To use this dataset:&#10;&#10;```python&#10;import tensorflow_datasets as tfds&#10;&#10;ds = tfds.load(&#x27;big_patent&#x27;, split=&#x27;train&#x27;)&#10;for ex in ds.take(4):&#10;  print(ex)&#10;```&#10;&#10;See [the guide](https://www.tensorflow.org/datasets/overview) for more&#10;informations on [tensorflow_datasets](https://www.tensorflow.org/datasets).&#10;&#10;" />
  <meta itemprop="url" content="https://www.tensorflow.org/datasets/catalog/big_patent" />
  <meta itemprop="sameAs" content="https://evasharma.github.io/bigpatent/" />
  <meta itemprop="citation" content="@misc{sharma2019bigpatent,&#10;    title={BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization},&#10;    author={Eva Sharma and Chen Li and Lu Wang},&#10;    year={2019},&#10;    eprint={1906.03741},&#10;    archivePrefix={arXiv},&#10;    primaryClass={cs.CL}&#10;}" />
</div>

# `big_patent`

Note: This dataset has been updated since the last stable release. The new
versions and config marked with
<span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>
are only available in the `tfds-nightly` package.

*   **Description**:

BIGPATENT, consisting of 1.3 million records of U.S. patent documents along with
human written abstractive summaries. Each US patent application is filed under a
Cooperative Patent Classification (CPC) code. There are nine such classification
categories: A (Human Necessities), B (Performing Operations; Transporting), C
(Chemistry; Metallurgy), D (Textiles; Paper), E (Fixed Constructions), F
(Mechanical Engineering; Lightning; Heating; Weapons; Blasting), G (Physics), H
(Electricity), and Y (General tagging of new or cross-sectional technology)

There are two features: - description: detailed description of patent. -
summary: Patent abastract.

*   **Homepage**:
    [https://evasharma.github.io/bigpatent/](https://evasharma.github.io/bigpatent/)

*   **Source code**:
    [`tfds.summarization.BigPatent`](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/summarization/big_patent.py)

*   **Versions**:

    *   **`2.1.0`** (default)
        <span class="material-icons" title="Available only in the tfds-nightly package">nights_stay</span>:
        Fix update to cased raw strings.
    *   `1.0.0`: No release notes.
    *   `2.0.0`: No release notes.

*   **Download size**: `Unknown size`

*   **Dataset size**: `Unknown size`

*   **Auto-cached**
    ([documentation](https://www.tensorflow.org/datasets/performances#auto-caching)):
    Unknown

*   **Splits**:

Split | Examples
:---- | -------:

*   **Features**:

```python
FeaturesDict({
    'abstract': Text(shape=(), dtype=tf.string),
    'description': Text(shape=(), dtype=tf.string),
})
```

*   **Supervised keys** (See
    [`as_supervised` doc](https://www.tensorflow.org/datasets/api_docs/python/tfds/load#args)):
    `('description', 'abstract')`

*   **Citation**:

```
@misc{sharma2019bigpatent,
    title={BIGPATENT: A Large-Scale Dataset for Abstractive and Coherent Summarization},
    author={Eva Sharma and Chen Li and Lu Wang},
    year={2019},
    eprint={1906.03741},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```

*   **Visualization**
    ([tfds.show_examples](https://www.tensorflow.org/datasets/api_docs/python/tfds/visualization/show_examples)):
    Not supported.

## big_patent/all (default config)

*   **Config description**: Patents under all categories.

## big_patent/a

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)a: Human Necessities

## big_patent/b

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)b: Performing Operations; Transporting

## big_patent/c

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)c: Chemistry; Metallurgy

## big_patent/d

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)d: Textiles; Paper

## big_patent/e

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)e: Fixed Constructions

## big_patent/f

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)f: Mechanical Engineering; Lightning; Heating; Weapons; Blasting

## big_patent/g

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)g: Physics

## big_patent/h

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)h: Electricity

## big_patent/y

*   **Config description**: Patents under Cooperative Patent Classification
    (CPC)y: General tagging of new or cross-sectional technology
