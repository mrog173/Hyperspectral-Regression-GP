# Genetic Programming for Hyperspectral Image Regression

This code is the Python implementation of the Genetic Programming (GP) implementation proposed in the paper "<i>Evolving Multispectral Sensor Configurations Using Genetic Programming for Estuary Health Monitoring</i>". This method evolves feature extractors to select key wavelengths and construct features from hyperspectral imaging, maximising the performance of support vector regression (SVR) models over generations. This method was originally evaluated using a dataset of sediment samples with corresponding organic matter content measurements.

For more details, see \<Publication pending\> and the corresponding dataset: [FigShare](http://doi.org/10.17608/k6.auckland.25546396).

## Installation

The code in this repository requires [DEAP](https://github.com/DEAP/deap), and uses various functions from [OpenCV](https://github.com/opencv/opencv), [SciPy](https://docs.scipy.org/doc/scipy/), and [Scikit-learn](https://scikit-learn.org/). The environment can be installed with anaconda using the <i>env.yaml</i> file.

```bash
conda env create --name <NAME> --file env.yaml
```

Visualising the output trees requires [PyGraphViz](https://pygraphviz.github.io/), if you encounter problems installing this, the config file can be edited to skip this operation, and the package can be ignored from the <i>env.yaml</i>.

## Example usage
To begin an evolutionary run: 

```bash
$ python -m Hyperspectral_GP.py <run_number> <config_file>
```

* Run number is used to specify the number of the run. This is helpful if you run many times and want to have unique numbers for each run.
* The config file located in the "Config/" folder is used to specify the arguments of the run. E.g., "<i>sediment_exp8.yml</i>".


### Custom config files
The evolutionary runs can be configured in the config files. The list of the arguments defined in the config files is found below:

<table>
  <tr>
    <th>Argument name</th>
    <th>Description</th>
  </tr>
  <tr>
    <th>attribute</th>
    <th>Which attribute to predict. "<i>Porosity</i>" and "<i>Organic Matter</i>" are supported.</th>
  </tr>
  <tr>
    <th>cache_table</th>
    <th>Whether to cache results to reduce the number of repeated evaluations (true/false).</th>
  </tr>
  <tr>
    <th>crossover_prob</th>
    <th>Probability that crossover is performed.</th>
  </tr>
    <tr>
    <th>mutation_prob</th>
    <th>Probability of a mutation operation being applied.</th>
  </tr>
  <tr>
    <th>mut_eph_prob</th>
    <th>Probability of mutating random random constants given that a mutation is applied (e.g., 0.25 of all mutations will only affect the leaf nodes). </th>
  </tr>
  <tr>
    <th>elitism_prob</th>
    <th>Percentage of the best individuals to copy directly over to the next population.</th>
  </tr>
  <tr>
    <th>fitness_function</th>
    <th>Which fitness function to use to govern the evolutionary process. "<i>MSE</i>", "<i>R<sup>2</sup></i>", and "<i>RMSE</i>" are supported.</th>
  </tr>
  <tr>
    <th>generations</th>
    <th>Number of generations to evolve individuals for.</th>
  </tr>
  <tr>
    <th>image_based</th>
    <th>Whether to evolve individuals based on the image or spectra data (true/false).</th>
  </tr>
  <tr>
    <th>initial_max_depth</th>
    <th>Initial maximum depth of initialised individuals.</th>
  </tr>
  <tr>
    <th>initial_min_depth</th>
    <th>Initial minimum depth of initialised individuals.</th>
  </tr>
  <tr>
    <th>maxDepth</th>
    <th>Maximum depth of individuals.</th>
  </tr>
  <tr>
    <th>pop_size</th>
    <th>Population size for evolutionary run.</th>
  </tr>
  <tr>
    <th>tournament_size</th>
    <th>Size of the tournament for tournament selection.</th>
  </tr>
  <tr>
    <th>save_tree</th>
    <th>Whether to save the best tree as a PDF (true/false). Requires PyGraphViz to draw the solution graph.</th>
  </tr>
  </table>

### Applying the model

The <i>run_model.py</i> file applys a final tree to the entire sediment dataset, training on the training set samples and then visualising each of the 30 quadrats (divided into 25 samples).


### Adapting to your own dataset

If you wish to use this with your own dataset, you can see an example of the dataloader in <i>dataloader_sediment.py</i>, which is called from the <i>import_data</i> function in the <i>helper_functions.py</i> file.

## License

This repository is released under the MIT license as found in the LICENSE file.

## Citation

If you find this repository useful, please consider giving it a citation:

```
<BibTeX pending publication>
```
